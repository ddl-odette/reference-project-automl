import configargparse
import yaml

def some_script(args):
    
        print(args.a)
        print(args.b)

if __name__ == "__main__":
    
    parser = configargparse.ArgumentParser(
        description="Test",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        # formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )
    
    parser.add(
        "--config",
        is_config_file=True, 
        help="config file path"
        )
    
    parser.add('--a',
               default=0,
               type=int,
               help='Number of GPUs')
    
    parser.add('--b', 
               default='pytorch',
               type=str,
               help='Backend library')
    
    args = parser.parse_args()
    
    print(args)
    
    some_script(args)


    