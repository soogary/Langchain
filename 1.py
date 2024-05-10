import numpy as np
import pandas as pd
import plotly.express as px

#generate random numbers from uniform distribution between 0 and 1

random_nums = np.random.rand(10)
print(random_nums)


#generate random numbers from a normal distribution
data = pf.Dataframe({'x':np.arange(30), 'y': np.random.randn(30)})


#create line chart
fig = px.line(data, x='x', y = 'y')

#show chart
fig.show()
