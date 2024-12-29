#!/usr/bin/env python3

# Solve all releases in the public index; compare to previous results if
# existing and report any changes in solvability of crates.

# Class holding crate name, version string, solvable and time to solve
import json
import os
import platform
import random
import subprocess
import time

# set to false to only test against stored values without saving
SAVING = True
ALR="alr-2.0.2"
STD_DEVS = 3.0 # 3sigma=99.73%, 2sigma=95.45%, 1sigma=68.27%
MIN_SAMPLES = 20 # to detect outliers
KEEP_SAMPLES = 20
TIMEOUT = 30

# Global to hold alr version
alr_version = ""


class Release:
    def __init__(self, name, version, solvable=None):
        self.name = name
        self.version = version
        self.solvable = solvable
        self.samples = []

    def milestone(self) -> str:
        return f"{self.name}={self.version}"

    def solve(self, max:int, regressions:set, progressions:set) -> bool:
        # Solve the crate and keep track of time needed
        start_time = time.time()
        try:
            p = subprocess.run([ALR,
                                "show",
                                f"{self.name}={self.version}",
                                "--solve"],
                                capture_output=True,
                                timeout=TIMEOUT)
            p.check_returncode()
            new_solvable = ("Dependencies cannot be met" not in p.stdout.decode())
        except subprocess.CalledProcessError:
            new_solvable = False
        except subprocess.TimeoutExpired:
            new_solvable = False
            print(f"  TIMEOUT after {TIMEOUT} seconds")

        elapsed = round(time.time() - start_time, 2)

        # Print results
        print(f"  Solvable: {new_solvable} in {elapsed:.2f} seconds "
              f"{'NOT SOLVABLE' if not new_solvable else ''}")

        # Compare to previous results
        if self.solvable is not None:
            if new_solvable != self.solvable:
                print(f"  CHANGE from {self.solvable} to {new_solvable}"
                      f" {'REGRESSION' if not new_solvable else 'PROGRESSION'}")
                self.solvable = new_solvable
                self.samples = []
                if new_solvable:
                    progressions.add(self.milestone())
                else:
                    regressions.add(self.milestone())
                return new_solvable
            else:
                # Check timings

                outlier = False

                if len(self.samples) > 1:
                    avg_time = sum([sample for sample in self.samples]) / len(self.samples)
                    std_dev = (sum([(sample - avg_time) ** 2 for sample in self.samples]) / len(self.samples)) ** 0.5
                    print(f"  Avg time: {avg_time:.2f} s, Std dev: {std_dev:.2f} s")

                    # Test if the new sample is an outlier
                    if len(self.samples) >= MIN_SAMPLES:
                        if abs(elapsed - avg_time) > STD_DEVS * std_dev:
                            outlier = True
                            good = elapsed < avg_time
                            print(f"  OUTLIER: {elapsed:.2f} s "
                                  f"{'GOOD' if good else 'BAD'} "
                                  f"({elapsed - avg_time:+.2f} s)")

                if not outlier:
                    # Add the new sample to the list
                    self.samples.append(elapsed)

                return not outlier
        else:
            self.solvable = new_solvable
            self.samples.append(elapsed)
            return True


    def path(self):
        return f"samples/{platform.node()}/{self.name}={self.version}.json"

    def load(self, max:int) -> bool:
        # Don't load if release already has samples
        if len(self.samples) > 0:
            return True

        # Load the release from a previous run
        if os.path.isfile(self.path()):
            with open(self.path(), "r") as f:
                data = json.load(f)
                self.solvable = data["solvable"]
                self.samples = data["samples"]
                if len(self.samples) > max:
                    self.samples = self.samples[-max:]
                return True
        else:
            return False

    def save(self):
        if not SAVING:
            return

        if len(self.samples) > KEEP_SAMPLES:
            self.samples = self.samples[-KEEP_SAMPLES:]

        # Create parent directory if it does not exist
        os.makedirs(os.path.dirname(self.path()), exist_ok=True)

        # Save the release to a file
        with open(self.path(), "w") as f:
            json.dump({"solvable": self.solvable,
                       "samples": self.samples}, f)



# Function that lists all releases in the public index
def list_releases(crate:str="") -> list:
    # Obtain release from `alr search`, which returns a json list of objects:
    json_releases = json.loads(subprocess.check_output
                               (["alr", "--format", "search", "--full", "--list"])
                               .decode())

    # Convert to list of Release objects
    releases = []
    for release in json_releases:
        if crate is not None and crate not in f'{release["name"]}={release["version"]}':
            continue
        releases.append(Release(release["name"], release["version"]))

    return releases


def plot(releases:list):
    import matplotlib.pyplot as plt

    # Filter out releases with no samples
    releases = [release for release in releases if len(release.samples) > 0]

    # Sort by mean time to solve
    releases.sort(key=lambda release: sum(release.samples) / len(release.samples))

    # Plot a box plot for each release
    fig, ax = plt.subplots()
    ax.set_title(f"Alire {alr_version}")
    ax.set_ylabel("Time to solve (s)")
    ax.set_xlabel("Release")
    ax.boxplot([release.samples for release in releases],
               labels=[release.milestone() for release in releases],
               vert=False)
    # ax.violinplot([release.samples for release in releases], showmeans=True)
    # ax.set_xticks(range(1, len(releases) + 1))
    # ax.set_xticklabels([release.milestone() for release in releases])
    plt.show()


def parse_args() -> dict:
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Solve all releases in the public index")
    parser.add_argument("--crate", help="Crate name/milestone to solve")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of rounds to solve each crate")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    args = parser.parse_args()

    return args


def main():

    # Obtain Alire version from `alr --version` and store in the global
    global alr_version
    alr_version = subprocess.check_output(["alr", "--version"]).decode("utf-8")

    args = parse_args()

    # List all releases in the public index
    print("Listing releases...")
    releases = list_releases(args.crate)

    if not args.plot:
        print(f"Running {args.rounds} rounds for {args.crate if args.crate else 'all crates'}")

    # Randomize list order to avoid bias from partial runs to some extent
    # TODO: save releases all at once at the end of a run
    random.shuffle(releases)

    # Keep track of regressions and progressions
    progressions = set()
    regressions = set()

    # If not saving, only test for one round against stored values
    if not SAVING:
        args.rounds = 1

    for _ in range(args.rounds):
        # For each release, solve it and compare to previous results
        for release in releases:
            print(f"{release.name}={release.version}")

            if release.load(KEEP_SAMPLES):
                print(f"  Loaded {len(release.samples)} samples")

            if args.plot:
                continue

            # Skip if already solved required samples
            if len(release.samples) >= KEEP_SAMPLES:
                print(f"  Already solved {KEEP_SAMPLES} samples, skipping")
                continue

            # Solve the release, keeping track of time needed
            if release.solve(KEEP_SAMPLES, regressions, progressions):
                release.save()

    if args.plot:
        plot(releases)
    else:
        # Print results
        print(f"Progressions: {len(progressions)}")
        for progression in progressions:
            print(f"  {progression}")
        print(f"Regressions: {len(regressions)}")
        for regression in regressions:
            print(f"  {regression}")


# Start of main script
if __name__ == "__main__":
    main()