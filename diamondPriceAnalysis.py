#The dataset we will be using contains information about diamonds like :
# 1. Carat
# 2. Cut
# 3. Color
# 4. Clarity
# 5. Depth
# 6. Table
# 7. Price
# 8. Size

#Let's import the necessary libraries and load the dataset

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go 

data = pd.read_csv("diamonds.csv")
print(data.head())

#The dataset contains an unnamed column, we will remove it since it's useless

data = data.drop("Unnamed: 0", axis=1)

#Let's start analyzing diamond prices. I will first analyze the relationship between the carat and the price of the diamond to see how the number of carats affects the price of a diamond:

figure = px.scatter(data_frame = data, x="carat", y="price", size="depth", color= "cut", trendline="ols")
figure.show()

#in the figure we can see a linear relationship between the number of carats and the price of a diamond. 
# It means higher carats result in higher prices. 

#Now I will add a new column to this dataset by calculating the size (length*width*depth) of a diamond. 

data["size"] = data["x"] * data["y"] * data["z"]
print(data) 

#Let's have a look at the relationship between the size of a diamond and its price:

figure = px.scatter(data_frame = data, x="size", y="price", size="size", color= "cut", trendline="ols")
figure.show()

#The figure concludes two features of diamons:

#1. Premium cut diamonds are relatively large than other diamonds.
#2. There's a linear relationship between the size of all tupes of diamonds and their prices

#Let's have a look at the prices of all the types of diamonds based on their colour:

fig = px.box(data, x="cut", y="price", color="color")
fig.show()

#We can see that the price depends on color and cut too. 

#Let's have a look at the prices of all types of diamonds based on clarity. 

fig = px.box(data, x="cut", y="price", color="clarity")
fig.show()

#Now we'll see the correlation between diamond prices and other features in the dataset:

correlation = data.corr()
print(correlation["price"].sort_values(ascending=False))

#Having analyzed the dataset, we will move to the task of predicting diamond prices using all the necessary information from the diamond price analysis done above. 

#I will convert the values of the cut column as the cut type of diamonds is a valuable feature to predict the price of a diamond. To use
#this column, we need to convert its categorical values into numerical values. 

data["cut"] = data["cut"].map({"Ideal": 1, "Premium": 2, "Good": 3, "Very Good": 4, "Fair": 5})

#Let's split the data into training and test sets:

from sklearn.model_selection import train_test_split
x = np.array(data[["carat", "cut", "size"]])
y = np.array(data[["price"]])
xtrain, xtest, ytrain ,ytest = train_test_split(x, y, test_size=0.10, random_state=42)

#Now I will train a machine learning model for the task of diamond price prediction:

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(xtrain, ytrain)

#Below is how we can use our machine learning model to predict the price of a diamond:

print("Diamond Price Prediction")
a = float(input("Carat Size: "))
b = int(input("Cut Type (Ideal: 1, Premium: 2, Good: 3, Very Good: 4, Fair: 5): "))
c = float(input("Size: "))
features = np.array([[a, b, c]])
print("Predicted Diamond's Price = ", model.predict(features))


#According to the diamond price analysis, we can say that the price and size of premium diamonds are higher than other types of diamonds.