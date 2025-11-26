import sys
import yaml
from scripts.training import training_loop

def run_with_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Convert YAML config into Click arguments
    args = []
    for key, value in config.items():
        print(key, value)
        if key == 'debug': 
            if value:
                args.append('--debug')
            else: 
                continue
        else:
            args.append(f'--{key}')
            args.append(str(value))

    # Invoke your training loop
    print("args:", args)
    training_loop(args, standalone_mode=False)

if __name__ == '__main__':
    # parse the yaml file path from command line arguments
    #print("sys.argv:", sys.argv)
    #if len(sys.argv) != 2:
    #    print("Usage: python training_with_config.py <path_to_yaml_config>")
    #    sys.exit(1)
    #yaml_path = sys.argv[1]
    yaml_path = 'configs/runs/resnet-learnable/e2cnn.yaml'
    if not yaml_path.endswith('.yaml') and not yaml_path.endswith('.yml'):
        print("Error: The provided file is not a YAML file.")
        sys.exit(1)
    run_with_config(yaml_path)

