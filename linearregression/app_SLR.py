import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score,classification_report,mean_squared_error
# page configuration
st.set_page_config("Linear Regression",layout="centered")
#load css
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
load_css("style.css")
#Title 
st.markdown("""
    <div class="card">
            <h1>Linear Regression</h1>
            <p>Predict<b>Tip Amount</b>from <b>Total Bill</b> using Linear Regression....</p>
    </div>
            """,unsafe_allow_html=True)
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Dataset preview")
st.dataframe(df.head())
st.markdown('</div>',unsafe_allow_html=True)
#prepare data
# prepare the data

x,y = df[["total_bill"]],df["tip"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train the model

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

# metrics

mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mae)
r2 = r2_score(y_test,y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)

# visualization

st.markdown('<div class = "card">',unsafe_allow_html=True)
st.subheader("Total Bill vs Tip Amount")
fig,ax = plt.subplots()
ax.scatter(x,y,alpha=0.5)
ax.plot(x,model.predict(scaler.transform(x)),color="red")
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Amount")
st.pyplot(fig)
st.markdown('<div>',unsafe_allow_html=True)
#performance
st.markdown('<div class = "card">',unsafe_allow_html=True)
st.subheader("model performance")
c1,c2=st.columns(2)
c1.metric("MAE",f"{mae:.2f}")
c2.metric("RMSE",f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric("r2",f"{r2:.2f}")
c4.metric("adj r2",f"{adj_r2:.3f}")
st.markdown('</div',unsafe_allow_html=True)
#m and c
st.markdown(f"""
        <div class="card">
            <h3> Model Intercept & coefficient</h3>
            <p><b> Co-effficient:</b>{model.coef_[0]:.3f}<br>
            <b>Intercept</b>{model.intercept_:.3f}</p>
            </div>
            """,unsafe_allow_html=True)
#prediction
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("predict tip amount")
bill=st.slider("Total bill",float(df.total_bill.min()),float(df.total_bill.max()),30.0)
tip=model.predict(scaler.transform([[bill]]))[0]
st.markdown(f'<div class="prediction-box">predict Tip:${tip:.2f}</div>',unsafe_allow_html=True)
st.markdown('</div>',unsafe_allow_html=True)