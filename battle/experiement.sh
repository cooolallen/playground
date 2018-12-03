agent1="ignore"
agent2="ignore"
agent3="ignore"
agent4="ignore"
nums_times=$1
# --record_json_dir=./battle

python battle/run_battle.py --agents=$agent1::null,$agent2::null,$agent3::null,$agent4::null --num_times=$nums_times --config=PommeTeamCompetition-v0