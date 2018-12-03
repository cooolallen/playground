agent1="ignore::null"
agent2="test::agents.SimpleAgent"
agent3="ignore::null"
agent4="test::agents.SimpleAgent"
nums_times=$1
# --record_json_dir=./battle

python battle/run_battle.py --agents=$agent1,$agent2,$agent3,$agent4 --num_times=$nums_times --config=PommeTeamCompetition-v0