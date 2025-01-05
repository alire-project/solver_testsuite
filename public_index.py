#!/usr/bin/env python3

# Solve all releases in the public index; compare to previous results if
# existing and report any changes in solvability of crates.

# Class holding crate name, version string, solvable and time to solve
import copy
import json
import matplotlib.pyplot as plt
import os
import platform
import random
import subprocess
import time

from tqdm import tqdm

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
    def __init__(self, name, version):
        self.name = name
        self.version = version
        self.samples = []
        self.solved = 0
        self.unsolved = 0

    def milestone(self) -> str:
        return f"{self.name}={self.version}"

    def solve(self) -> bool:
        # Solve the crate and keep track of time needed
        start_time = time.time()
        timeout = False
        error = False
        try:
            p = subprocess.run([ALR,
                                "show",
                                f"{self.name}={self.version}",
                                "--solve"],
                                capture_output=True,
                                timeout=TIMEOUT)
            p.check_returncode()
            solvable = ("Dependencies cannot be met" not in p.stdout.decode())
        except subprocess.CalledProcessError:
            solvable = False
            error = True
        except subprocess.TimeoutExpired:
            solvable = False
            timeout = True
            print(f"  TIMEOUT after {TIMEOUT} seconds")

        elapsed = round(time.time() - start_time, 2)

        # Update solved/unsolved counts
        if solvable:
            self.solved += 1
        else:
            self.unsolved += 1

        self.samples.append(elapsed)

        # Print results
        print(f"  Solvable: {solvable} in {elapsed:.2f} seconds "
              f"{'NOT SOLVABLE' if not solvable else ''}"
              f"{' (error)' if error else ''}"
              f"{' (timeout)' if timeout else ''}")

        return solvable


    def clear_samples(self):
        self.samples = []
        self.solved = 0
        self.unsolved = 0

    def compute_stats(self):
        if len(self.samples) > 0:
            self.average = sum(self.samples) / len(self.samples)
            self.std_dev = \
                (sum([(sample - self.average) ** 2
                      for sample in self.samples]) / len(self.samples)) ** 0.5
        else:
            self.average = None
            self.std_dev = None

    def solvable(self):
        return self.solved / (self.solved + self.unsolved)

    def solvable_img(self):
        if self.solved + self.unsolved == 0:
            return "??"
        elif self.solved == 0:
            return "NS"
        elif self.unsolved == 0:
            return "OK"
        else:
            return f"{self.solved/(self.solved + self.unsolved):.1f}"

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
                try:
                    data = json.load(f)
                except:
                    print(f"Error loading {self.path(tag)}")
                    raise
                self.samples = data["samples"]
                if len(self.samples) > max:
                    self.samples = self.samples[-max:]

                # For back-compatibility, convert solvable if existing to
                # solved/unsolved counts
                if "solvable" in data:
                    self.solved = len(self.samples) if data["solvable"] else 0
                    self.unsolved = len(self.samples) - self.solved
                else:
                    self.solved = data["solved"]
                    self.unsolved = data["unsolved"]

            # Drop any samples with suspicious timing (probably taken with
            # interruptions) that have twice or more the timeout value
            self.samples = [sample for sample in self.samples
                            if sample < 2 * TIMEOUT]

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
            json.dump({"solved": self.solved,
                       "unsolved": self.unsolved,
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
    releases = []
    for release in tqdm(json_releases, desc="Filtering out independent releases"):

        # Print progress using a nice library, since we know the total releases


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


def plot(releases:list, baseline:list=None, include:str=""):

    # Filter out releases with no samples
    releases = [release for release in releases if len(release.samples) > 0]

    # Filter out releases not in --plot-include filter
    if include:
        print(f"Filtering with --plot-include={include}")
        releases = [release for release in releases
                    if include in release.solvable_img()]
        if baseline:
            baseline = [release for release in baseline
                        if include in release.solvable_img()]

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

    # Bail out if no releases remain
    if len(releases) == 0:
        print("No releases left to plot")
        return

    # Sort by mean time to solve
    releases.sort(key=lambda release: release.average
                  if release.average is not None else float("inf"))

    # Prepare data for plotting
    release_samples = [release.samples for release in releases]
    release_labels = [f"{release.milestone()} "
                      f"({release.solvable_img()})"
                      for release in releases]
    if baseline is not None:
        baseline_labels = ["baseline "
                        f"({baseline[release.milestone()].solvable_img()})"
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

    # Check if the lengths match
    if len(release_samples) != len(release_labels):
        raise ValueError("The number of release samples and labels must be"
                         " the same: "
                         f"{len(release_samples)} != {len(release_labels)}")

    # Plot the boxplots for releases
    ax.boxplot(release_samples, tick_labels=release_labels, vert=False,
               positions=positions,
               patch_artist=True, boxprops=dict(facecolor="lightblue"))

    # Overlay the boxplots for baseline
    if baseline is not None:
        ax.boxplot(baseline_samples, tick_labels=baseline_labels, vert=False,
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
    print(f"Solvable: {len([release for release in releases if release.solvable >= 0.5])}")
    print(f"Unsolvable: {len([release for release in releases if release.solvable < 0.5])}")


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
    parser.add_argument("--plot-include", type=str, default=None,
                        help="Substring to look for in solving status")

    parser.add_argument("--prune", action="store_true", help="Prune outliers")

    args = parser.parse_args()

    if args.alr is not None:
        global ALR
        ALR = args.alr

    return args


def main():

    start = time.time()

    args = parse_args()

    # Obtain Alire version from `alr --version` and store in the global
    global ALR_VERSION
    ALR_VERSION = subprocess.check_output([ALR, "--version"]).decode("utf-8").strip()
    # The version is actually the part after the space
    ALR_VERSION = ALR_VERSION.split(" ")[1]

    # Early warn if tag differs from alr version and ask to continue
    if args.solve and args.tag != ALR_VERSION:
        print(f"Warning: tag ({args.tag}) differs from alr version ({ALR_VERSION})")
        if input("Continue? (y/n) ").lower() != "y":
            return

    # List all releases in the public index
    print("Listing releases...")
    releases = list_releases(args.crate)

    print(f"Loading releases ({args.tag})...")
    releases = load_releases(releases, args.tag)

    baseline = None
    if args.compare:
        print(f"Loading baseline ({args.compare})...")
        baseline = load_releases(releases, args.compare)

    if args.solve:
        print(f"Running {args.rounds} rounds for {args.crate if args.crate else 'all crates'}")

    # If not saving, only test for one round against stored values
    if not SAVING:
        args.rounds = 1

    if args.plot:
        plot(releases, baseline, include=args.plot_include)
    elif args.prune:
        for release in releases:
            print(f"{release.name}={release.version}: ", end="")
            release.drop_outliers()
            release.save(args.tag)
        report(releases)
    elif args.solve:
        # Randomize list order to avoid bias from partial runs to some extent
        random.shuffle(releases)

        for _ in range(args.rounds):

            min_samples = min([len(release.samples) for release in releases])

            # For each release, solve it and compare to previous results
            for release in releases:
                print(f"{release.name}={release.version} "
                      f"({len(release.samples)} samples)")

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
                release.solve()
                release.save(args.tag)
    else:
        # Report num of releases and avg number of samples
        print(f"Tag: {args.tag}")
        report(releases)
        print("No action to perform")

    # Report finish time and elapsed time
    print(f"Finished at {time.strftime('%H:%M:%S')}, "
          f"elapsed: {time.time() - start:.2f} seconds")

# Start of main script
if __name__ == "__main__":
    main()