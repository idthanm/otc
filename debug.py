import subprocess

subprocess.call(['python', '-um', 'dopamine.discrete_domains.train', '--base_dir=.', '--gin_files=.\\dopamine\\agents\\dqn\\configs\\dqn.gin'])