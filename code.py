# Install TensorFlow Decision Forests
!pip install tensorflow_decision_forests
# Load TensorFlow Decision Forests
import tensorflow_decision_forests as tfdf

# Load the training dataset using pandas
import pandas
train_df = pandas.read_csv("/content/drive/MyDrive/TRAININGONE1.csv")

d = {'(empty)   Benign   -': 0, '(empty)   Malicious   PartOfAHorizontalPortScan': 1,'(empty)   Malicious   Attack':1,'(empty)   Malicious   C&C':1}
train_df['resp_ip_bytes'] = train_df['resp_ip_bytes'].map(d)
val = train_df[(train_df.resp_ip_bytes!=1) & (train_df.resp_ip_bytes!=0) ]
print(val)

# Convert the pandas dataframe into a TensorFlow dataset
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="resp_ip_bytes")


# Train the model
model = tfdf.keras.RandomForestModel()
model.fit(train_ds)

# Load the testing dataset
test_df = pandas.read_csv("/content/drive/MyDrive/TESTINGONE1.csv")
#d = {'(empty)   Benign   -': 1, '(empty)   Malicious   PartOfAHorizontalPortScan': 2,'(empty)   Malicious   Attack':2,'(empty)   Malicious   C&C':2}
test_df['resp_ip_bytes'] = test_df['resp_ip_bytes'].map(d)
# Convert it to a Tens
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="resp_ip_bytes")

# Evaluate the model
model.compile(metrics=["accuracy"])
print(model.evaluate(test_ds))

model.save("/content/drive/MyDrive/project/my_first_model")

tfdf.model_plotter.plot_model_in_colab(model, tree_idx=0)