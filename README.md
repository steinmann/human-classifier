### Design
* a simple command line interfacte to feed data, return model accuracy and save the model
* uses default xgboost binary classifier
* excludes heroes that lack info on race and superpowers
* uses all properties except name as features
* categorical attributes are one hot encoded
* uses five fold cross validation to determine the model accuracy 

### Usage
usage: train.py [-h] --info [INFO] --power [POWER] [--model [MODEL]]

-h, --help       show this help message and exit

--info [INFO]    csv file containing general superhero info

--power [POWER]  csv file containing info on superhero powers

--model [MODEL]  name of the model (optional)
