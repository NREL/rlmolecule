#! /bin/bash
<< 'Information'
 Function conda_auto_env enables the user to automatically activate an environment once they enter into
 a project directory, assuming an <environment_name>.yaml already exists in the directory. If that is not
 the case, then conda_auto_env will automatically create one based on the existing yaml file.

 function conda_auto_env() {
  if [ -e "molecule_environment.yml" ]; then
    # echo "molecule_environment.yml file found"
    ENV=$(head -n 1 molecule_environment.yml | cut -f2 -d ' ')
    # Check if you are already in the environment
    if [[ $PATH != *$ENV* ]]; then
      # Check if the environment exists
      conda activate $ENV
      if [ $? -eq 0 ]; then
        :
      else
        # Create the environment and activate
        echo "Conda env '$ENV' doesn't exist."
        conda env create -q
        conda activate $ENV
      fi
    fi
  fi
}

export PROMPT_COMMAND=conda_auto_env
Information

conda activate path-to-env-eagle

num_workers=2

for (( i=1; i<=$num_workers; i++ ))
do
    python rollout.py --id "$i"  &
done