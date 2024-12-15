import re


def import_script():
    file_path = "/Users/Christopher/Downloads/Uptuna 2 Final.txt"
    with open(file_path, 'r') as file:
        file_content = file.read()
        return file_content


def process_episodes(episodes):
    scores = []
    for episode in episodes:
        # Use regex to extract score, handling potential variations
        match = re.search(r'score: \[(\d+)\]', episode)
        if match:
            scores.append(int(match.group(1)))

    last_episode = episodes[-1]
    params = last_episode.split("{")[1].split("Best")[0]

    return {
        'Parameters': params,
        'Mean Reward': round(sum(scores) / len(scores), 2) if scores else 0,
        'Median Reward': sorted(scores)[len(scores) // 2] if scores else 0,
        'Episode Rewards': scores
    }


def main():
    file_content = import_script()

    # Split trials
    trials_list = file_content.split("Environment is ready")
    trials_list = trials_list[1:]
    processed_trials = []
    for trial in trials_list:
        lines = trial.split("\n")

        episodes = [line for line in lines if "episode" in line.lower() or "finished with value:" in line.lower()]

        processed_trial = process_episodes(episodes)
        processed_trials.append(processed_trial)

    for i, trial in enumerate(processed_trials):
        print(
            f"Trial {i}: Parameters: {trial['Parameters']} Mean Reward (from episodes): {trial['Mean Reward']} Median Reward (from episodes): {trial['Median Reward']} Episode Rewards: {trial['Episode Rewards']}")

    return processed_trials


if __name__ == '__main__':
    main()