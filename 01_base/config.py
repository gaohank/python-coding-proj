import configparser

config = configparser.ConfigParser()

config.read('../resource/config.ini')
print(config.sections())
print(config['DEFAULT']['Compression'])
print(config['secret.server.com']['Port'])
print(config['Paths']['home_dir'])
print(config['Paths']['my_dir'])
print(config['Paths']['log_dir'])
