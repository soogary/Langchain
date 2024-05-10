import numpy as np
import pandas as pd
import plotly.express as px
import os
#Load API keys
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


#Using Numpy
#generate random numbers from uniform distribution between 0 and 1

random_nums = np.random.rand(10)
print(random_nums)


#Using Pandas
#generate random numbers from a normal distribution
data = pd.DataFrame({'x':np.arange(30), 'y': np.random.randn(30)})

#create line chart
fig = px.line(data, x='x', y = 'y')

#show chart
fig.show()



#os.environ.get('OPENAI_API_KEY')


from 