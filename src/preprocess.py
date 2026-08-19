import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load():
    # Idenitify features and target
    FEATURES = ["Absolute Magnitude", "Est Dia in KM(min)", "Est Dia in KM(max)", 
                "Relative Velocity km per sec", "Miss Dist.(kilometers)", "Orbit Uncertainity", 
                "Minimum Orbit Intersection", "Jupiter Tisserand Invariant", "Eccentricity", 
                "Semi Major Axis", "Inclination", "Asc Node Longitude", "Orbital Period", 
                "Perihelion Distance", "Perihelion Arg","Aphelion Dist", "Mean Anomaly", 
                "Mean Motion"]

    TARGET = "Hazardous"

    # Load into dataframes
    df = pd.read_csv("../data/nasa.csv")

    x = df[FEATURES]
    y = df[TARGET].astype(int)

    return x, y


def split_scale(x, y):

    # Split into train, validate and test groups - 70-15-15 split
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=17, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=17, stratify=y_temp)

    # Normalize values in each split based on mean and std.dev from the train set
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train.values, x_val_scaled, y_val.values, x_test_scaled, y_test.values 

def preprocess():
    '''Loads data from data/nasa.csv. Splits data into training (70%), validation (15%), and test (15%) sets.
    Returns list of numpy arrays in the following order:
    x_train, y_train, x_val, y_val, x_test, y_test'''

    x, y = load()
    x_train, y_train, x_val, y_val, x_test, y_test = split_scale(x, y)
    return [x_train, y_train, x_val, y_val, x_test, y_test]