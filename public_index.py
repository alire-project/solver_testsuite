#!/usr/bin/env python3

# Solve all releases in the public index; compare to previous results if
# existing and report any changes in solvability of crates.

# Class holding crate name, version string, solvable and time to solve
import copy
import json
import os
import platform
import random
import subprocess
import time

# set to false to only test against stored values without saving
SAVING = True

STD_DEVS = 3 # 3sigma=99.73%, 2sigma=95.45%, 1sigma=68.27%
PLOT_DEVS = 1.5
# We want larger dev to reject outliers; meanwhile, for plotting we want to see
# smaller differences.

MIN_SAMPLES_FOR_OUTLIERS = 10 # Number of samples before we do any filtering
MAX_SAMPLES = 100 # max number of samples to keep for each release
TIMEOUT = 30 # seconds after which the search is aborted
FULL_LIST = False # set to True to solve all releases in a crate instead of just the latest one

ALR="alr-not-supplied"
ALR_VERSION = ""


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
                self.samples.append(elapsed)

                return True
        else:
            self.solvable = new_solvable
            self.samples.append(elapsed)
            return True


    def clear_samples(self):
        self.samples = []

    def compute_stats(self):
        if len(self.samples) > 0:
            self.average = sum(self.samples) / len(self.samples)
            self.std_dev = \
                (sum([(sample - self.average) ** 2
                      for sample in self.samples]) / len(self.samples)) ** 0.5
        else:
            self.average = None
            self.std_dev = None

    def drop_outliers(self):
        self.compute_stats()

        if len(self.samples) < MIN_SAMPLES_FOR_OUTLIERS:
            print(f"not enough samples ({len(self.samples)}) to drop outliers")
            return

        old_len = len(self.samples)
        if self.average is not None:
            self.samples = [sample for sample in self.samples
                            if abs(sample - self.average) <= STD_DEVS * self.std_dev]

        print(f"dropped {old_len - len(self.samples)} outliers ("
              f"{(old_len - len(self.samples)) * 100 / old_len:.1f}% of "
              f"{len(self.samples)} samples)")

    def path(self, tag:str):
        return \
            f"samples/{platform.node()}/" + \
            ("current/" if tag == "" else f"{tag}/") + \
            f"{self.name}={self.version}.json"

    def load(self, tag:str, max:int=999999) -> bool:
        # Don't load if release already has samples
        if len(self.samples) > 0:
            return True

        # Load the release from a previous run
        if os.path.isfile(self.path(tag)):
            with open(self.path(tag), "r") as f:
                data = json.load(f)
                self.solvable = data["solvable"]
                self.samples = data["samples"]
                if len(self.samples) > max:
                    self.samples = self.samples[-max:]
                return True
        else:
            return False

    def save(self, tag:str):
        if not SAVING:
            return

        if len(self.samples) > MAX_SAMPLES:
            self.samples = self.samples[-MAX_SAMPLES:]

        # Create parent directory if it does not exist
        os.makedirs(os.path.dirname(self.path(tag)), exist_ok=True)

        # Save the release to a file
        with open(self.path(tag), "w") as f:
            json.dump({"solvable": self.solvable,
                       "samples": self.samples}, f)


def load_releases(releases:list, tag:str) -> list:
    # Load the releases from a previous run. If a version is specified, load
    # from that specific location, otherwise load from the default location.
    result = copy.deepcopy(releases)

    # Clean samples to force reload in the clones
    for release in result:
        release.clear_samples()

    for release in result:
        if release.load(max=MAX_SAMPLES, tag=tag):
            print(f"   {release.name}={release.version} "
                f"{'solvable' if release.solvable else 'not solvable'} "
                f"({len(release.samples)} samples)")

    return result


# Function that lists all releases in the public index
def list_releases(crate:str="") -> list:
    # Obtain release from `alr search`, which returns a json list of objects:
    args = ["alr", "--format", "search", "--list"]
    if FULL_LIST:
        args.append("--full")
    json_releases = json.loads(subprocess.check_output(args).decode())

    # Convert to list of Release objects
    print("Filtering", end="", flush=True)
    releases = []
    for release in json_releases:
        print(".", end="", flush=True)
        if crate is not None and crate not in f'{release["name"]}={release["version"]}':
            continue
        release = Release(release["name"], release["version"])

        # Keep only if it has dependencies (no point in solving otherwise)
        if "Dependencies (direct):" in subprocess.run(["alr",
                                                       "show",
                                                       release.milestone()],
                                                      capture_output=True).stdout.decode():
            releases.append(release)

    print()
    return releases


def compute_stats(releases:list):
    # Compute statistics for each release
    for release in releases:
        if len(release.samples) > 0:
            release.average = sum(release.samples) / len(release.samples)
            release.std_dev = \
                (sum([(sample - release.average) ** 2
                      for sample in release.samples]) / len(release.samples)) ** 0.5
        else:
            release.average = None


def plot(releases:list, baseline:list=None):
    import matplotlib.pyplot as plt

    # Filter out releases with no samples
    releases = [release for release in releases if len(release.samples) > 0]

    # Convert baseline to dictionary for simpler lookup
    if baseline is not None:
        compute_stats(baseline)
        baseline = {release.milestone(): release for release in baseline}

    compute_stats(releases)

    # Filter out releases with average in the baseline average +/- 3 sigma
    old = len(releases)
    if baseline is not None:
        releases = [release for release in releases
                    if release.milestone() not in baseline
                    or baseline[release.milestone()].average is None
                    or abs(release.average - baseline[release.milestone()].average)
                    > PLOT_DEVS * (release.std_dev + baseline[release.milestone()].std_dev)]
        print(f"Filtered out {old - len(releases)} releases within statistical bounds")

    # Sort by mean time to solve
    releases.sort(key=lambda release: release.average
                  if release.average is not None else float("inf"))

    # Prepare data for plotting
    release_samples = [release.samples for release in releases]
    release_labels = [f"{release.milestone()} "
                      f"({'OK' if release.solvable else 'NS'})"
                      for release in releases]
    baseline_labels = ["baseline "
                       f"({'OK' if baseline[release.milestone()].solvable else 'NS'})"
                       for release in releases]

    # Calculate positions for the boxplots
    positions = list(range(1, len(release_samples) + 1))
    baseline_positions = [pos - 0.2 for pos in positions]  # Offset baseline positions

    if baseline is not None:
        baseline_samples = [baseline[release.milestone()].samples
                            if release.milestone() in baseline
                            else []
                            for release in releases]
    else:
        baseline_samples = [[] for _ in releases]

    # Plot a box plot for each release (if no baseline) or with a significant deviation
    fig, ax = plt.subplots()
    ax.set_title(f"Alire {ALR_VERSION}")
    ax.set_ylabel("Time to solve (s)")
    ax.set_xlabel("Release")

    ALPHA=0.1

    # Add red background for cases where baseline is solvable but current release is not
    for i, release in enumerate(releases):
        if baseline is not None and release.milestone() in baseline and baseline[release.milestone()].solvable and not release.solvable:
            ax.axhspan(positions[i] - 0.5, positions[i] + 0.3, color='red', alpha=0.2)

    # Likewise with a green background for cases where baseline is not solvable
    # but current release is
    for i, release in enumerate(releases):
        if baseline is not None and release.milestone() in baseline and not baseline[release.milestone()].solvable and release.solvable:
            ax.axhspan(positions[i] - 0.5, positions[i] + 0.3, color='green', alpha=0.1)

    # Likewise with a blue background for cases where the average has improved
    # from the baseline
    for i, release in enumerate(releases):
        if baseline is not None and release.milestone() in baseline and release.average is not None and baseline[release.milestone()].average is not None and release.average < baseline[release.milestone()].average:
            ax.axhspan(positions[i] - 0.5, positions[i] + 0.3, color='cyan', alpha=0.1)

    # Likewise with a yellow background for cases where the average has
    # worsened
    for i, release in enumerate(releases):
        if baseline is not None and release.milestone() in baseline and release.average is not None and baseline[release.milestone()].average is not None and release.average > baseline[release.milestone()].average:
            ax.axhspan(positions[i] - 0.5, positions[i] + 0.3, color='yellow', alpha=0.1)

    # Plot the boxplots for releases
    ax.boxplot(release_samples, labels=release_labels, vert=False,
               positions=positions,
               patch_artist=True, boxprops=dict(facecolor="lightblue"))

    # Overlay the boxplots for baseline
    if baseline is not None:
        ax.boxplot(baseline_samples, labels=baseline_labels, vert=False,
                   positions=baseline_positions,
                   patch_artist=True, boxprops=dict(facecolor="lightgreen"))

    # ax.violinplot([release.samples for release in releases], showmeans=True)
    # ax.set_xticks(range(1, len(releases) + 1))
    # ax.set_xticklabels([release.milestone() for release in releases])
    plt.show()


def report(releases:list):
    # Report num of releases and avg number of samples
    print(f"Releases: {len(releases)}")
    print(f"Max samples: {max([len(release.samples) for release in releases])}")
    print(f"Min samples: {min([len(release.samples) for release in releases])}")
    print(f"Avg samples: {sum([len(release.samples) for release in releases]) / len(releases):.1f}")
    print(f"Total samples: {sum([len(release.samples) for release in releases])}")
    print(f"Solvable: {len([release for release in releases if release.solvable])}")
    print(f"Unsolvable: {len([release for release in releases if not release.solvable])}")


def parse_args() -> dict:
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Solve all releases in the public index")

    parser.add_argument("--solve", action="store_true", help="Plot results")
    parser.add_argument("--tag", type=str, default="", required=True,
                        help="Tag to use for the results")
    parser.add_argument("--crate", help="Crate name/milestone to solve")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of rounds to solve each crate")
    parser.add_argument("--alr", type=str, default="alr",
                        help="Name of the alr executable")

    parser.add_argument("--plot", action="store_true", help="Plot results")
    parser.add_argument("--compare", type=str, default="",
                        help="Compare to given version")

    parser.add_argument("--prune", action="store_true", help="Prune outliers")

    args = parser.parse_args()

    if args.alr is not None:
        global ALR
        ALR = args.alr

    return args


def main():

    # Obtain Alire version from `alr --version` and store in the global
    global ALR_VERSION
    ALR_VERSION = subprocess.check_output(["alr", "--version"]).decode("utf-8").strip()
    # The version is actually the part after the space
    ALR_VERSION = ALR_VERSION.split(" ")[1]

    args = parse_args()

    # List all releases in the public index
    print("Listing releases...")
    releases = list_releases(args.crate)

    print(f"Loading releases ({args.tag})...")
    releases = load_releases(releases, args.tag)

    if args.compare:
        print(f"Loading baseline ({args.compare})...")
        baseline = load_releases(releases, args.compare)

    if args.solve:
        print(f"Running {args.rounds} rounds for {args.crate if args.crate else 'all crates'}")

    # Keep track of regressions and progressions
    progressions = set()
    regressions = set()

    # If not saving, only test for one round against stored values
    if not SAVING:
        args.rounds = 1

    if args.plot:
        plot(releases, baseline)
    elif args.prune:
        for release in releases:
            print(f"{release.name}={release.version}: ", end="")
            release.drop_outliers()
            release.save(args.tag)
        report(releases)
    elif args.solve:
        # Warn if tag differs from alr version and ask to continue
        if args.tag != ALR_VERSION:
            print(f"Warning: tag ({args.tag}) differs from alr version ({ALR_VERSION})")
            if input("Continue? (y/n) ").lower() != "y":
                return

        # Randomize list order to avoid bias from partial runs to some extent
        random.shuffle(releases)

        for _ in range(args.rounds):

            min_samples = min([len(release.samples) for release in releases])

            # For each release, solve it and compare to previous results
            for release in releases:
                print(f"{release.name}={release.version}")

                # Skip if more samples than previous release (to equalize) but
                # Allow chance to run so some progress is made. Eventually the
                # lagging ones should catch up. Use 50% chance for this.
                if len(release.samples) > min_samples and random.random() > 0.5:
                    print(f"  Excess {len(release.samples) - min_samples} samples, skipping")
                    continue

                # Skip if already solved required samples
                if len(release.samples) >= MAX_SAMPLES:
                    print(f"  Already solved {MAX_SAMPLES} samples, skipping")
                    continue

                # Solve the release, keeping track of time needed
                if release.solve(MAX_SAMPLES, regressions, progressions):
                    release.save(args.tag)

            # Print results
            print(f"Progressions: {len(progressions)}")
            for progression in progressions:
                print(f"  {progression}")
            print(f"Regressions: {len(regressions)}")
            for regression in regressions:
                print(f"  {regression}")
    else:
        # Report num of releases and avg number of samples
        print(f"Tag: {args.tag}")
        report(releases)
        print("No action to perform")


# Start of main script
if __name__ == "__main__":
    main()