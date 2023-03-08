# Session goals

1. Deploy an Azure Databricks workspace
2. Connect to Databricks clusters from DSS
3. Run some example recipes
4. Train a model in Databricks & store as an MLflow model object
5. Import this model into DSS

This is probably too much to do in one session, so if you have never used a Databricks connection, start from step 1. If you familiar with connecting to Databricks, start from step 4.

# References

[Databricks Connector article](https://analytics.dataiku.com/projects/DATAOPSWIKI/wiki/726/Databricks%20connector%20(v11.2))

# Steps

## 1 - Deploy an Azure Databricks workspace

### 1.1 - Create an Azure Databricks workspace

- Log into the Azure console, *Dataiku Engineering* subscription
- Create an Azure Databricks workspace with the following params
  - Resource group: databricks-demo
  - Workspace name: something unique to you
  - Region: uk south
  - Pricing tier: Premium, to allow testing of SQL warehouse functionality
  - Leave all other tabs (networking, encryption) as default

This should take about 3 minutes. When this is complete, go to the Azure workspace by clicking the URL of the newly created workspace in the Azure console, and clicking through all the sign in components. This should take you to the Databricks UI. 

### 1.2 - Create a *Data Science & Engineering* cluster
We need to create a DS&E cluster as this is the object that we will connect to from DSS.

When you enter the Databricks UI, expand the menu on the far left of the screen. By default you'll enter the DS&E platform (can be select from dropdown at the top left of the UI). From the menu on the left, select *Compute* and then click *Create compute* to create a Databricks cluster, with the following settings

- Policy: Unrestricted (so the following settings are consistent and present in your UI)
- Single node: Tick the single node box as we're just testing so there's no need for a real cluster
- Access mode: Leave as *Single User* and leave Single user access as your user
- Performance
  - Databricks runtime version: Change this to the latest **non-GPU** ML version (e.g. 12.1 ML, the one without GPUs). We need the ML version to get pre-installed ML packages, used in creating our MLflow model
  - Node type: Leave as smallest machine
  - Terminate after: Leave this or set this at some value >60 to avoid it turning off during the session

Now create the cluster. This will take ~5 mins so progress to the next step without waiting. Before leaving this page, click the *Advanced options* expander at the bottom of the screen > *JDBC/ODBC*. Record the *Server Hostname*, *port*, *HTTP Path* to be used in the Databricks connection.


### 1.3 - Start the default *SQL* warehouse
We also want to test connecting to a Databricks SQL warehouse endpoint. For this we need to start a SQL warehouse, in addition to the DS&E cluster we created.

To test connecting to a Databricks SQL warehouse, we'll also start the default cluster now. On the dropdown on the top left, navigate from *Data Science & Engineering* to *SQL* and then in the menu bar go to *SQL Warehouses*. 

Start the already present *Starter warehouse*. To get the connection details click into the cluster > *Connection details* and record the *Server Hostname*, *port*, *HTTP Path* as before


## 2 - Connect to a Databricks cluster from DSS

Log into the prepared [DSS instance](https://databricks-demo.fe-az.dkucloud-dev.com). This instance has the following pre-created connections that we will use:
- *common-adls*: An ADLS connection to pre-existing Storage Account [databricksdemodss](https://portal.azure.com/#@dkudemos.onmicrosoft.com/resource/subscriptions/82852abb-55ae-44d8-9bac-4632a4173215/resourcegroups/databricks-demo/providers/Microsoft.Storage/storageAccounts/databricksdemodss/overview) and container [databricks-demo](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2F82852abb-55ae-44d8-9bac-4632a4173215%2Fresourcegroups%2Fdatabricks-demo%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fdatabricksdemodss/path/databricks-demo/etag/%220x8DB08537346DC25%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) using an access key
- *common-adls-sas*: An ADLS connection to the same location but with an SAS token as well

Note that you can't see these connections because you ain't an admin, but they exist and you can use them.

### 2.1 - Create a Databricks DS&E connection

We now create an example Databricks connection, connecting the DS&E cluster. Go to connections and create a connection of type databricks, calling it something unique to you to avoid overlap with other users. 

For authentication we will use per-user OAuth. This requires setting a registered app in Azure, however not all FEs appear to have permissions to give the app *AzureDatabricks > Delegated > user_impersonation* permission. Use this pre-created [registered app](https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationMenuBlade/~/Credentials/appId/ed5f47b2-16d8-4f3b-abb2-cfe9337a5705/isMSAApp~/false) which has this permission and redirect URL already set. Create a new client secret for yourself (noting it down since we'll reuse it later). We'll refer to this later as `<YOUR_CLIENT_SECRET>`.

Enter the following params for the connection (leaving everything else default)
- Basic params
  - Host: Recorded when creating the DS&E cluster
  - Port: Recorded when creating the DS&E cluster
  - HTTP path: Recorded when creating the DS&E cluster
  - Auth type: *OAuth*
  - Client id: *ed5f47b2-16d8-4f3b-abb2-cfe9337a5705*
  - Client secret: Your secret you generated above
  - OAuth authorization endpoint: *https://login.microsoftonline.com/3ceb0d29-d7de-4204-b431-3f9f8edb2106/oauth2/v2.0/authorize*
  - Oauth token endpoint: *https://login.microsoftonline.com/3ceb0d29-d7de-4204-b431-3f9f8edb2106/oauth2/v2.0/token*
- Advanced params
  - Automatic fast-write: tick box
  - Automatic fast-write connection: *common-adls*
  - Path in connection: *databricks_tmp*
- Credentials
  - Credentials mode: *Per user*

Now go and create a connection credential for this connection, and then return and test the connection.

### 2.2 - Create a Databricks SQL connection

We also want to test interacting with the SQL warehouse, requiring a separate connection. Repeat the steps from above, creating a new connection. Everything is the same with the exception of the connection basic params, where the *HTTP path* will be different, so use the value that we recorded when setting up the SQL cluster. Test this connection works after adding a personal credential.

## 3 - Run some example recipes

To test the functionality of these connections, run some recipes covering the various key components:

1. In-database processing
2. Fast-write via ADLS
3. ADLS-Databricks direct sync

First create a new DSS project on this instance, with a personalised name to avoid clashes.

### 3.1 - Run some example DS&E recipes

Inside the project import one of the sample tables present in the DS&E Databricks connection (e.g. sample.nyctaxi.trips). You may need to manually detect and save the schema. Off of this dataset run the following recipes, always using the DS&E Databricks connection:
1. An in-database visual recipe back to the same connection to test SQL pushdown (ensure the SQL engine is selected)
2. A Python recipe back to the same connection to test the fast-write. Check the job logs for `fastpath` to check this is happening (though if it isn't, your job will not take several hours)
3. Sync the data to a Parquet dataset in connection *common-adls* using the *Databricks to Azure* engine and then sync back to Databricks usign the *Azure to Databricks* engine. 

These should all work given the above setup.

### 3.2 - Run some example SQL recipes

Repeat the above steps but now using the SQL Databricks connection in place of the DS&E connection (including re-importing the starting dataset). The in-database recipe should work without alteration, however both the Python recipe and the sync to ADLS will fail (the sync from ADLS back to Databricks would also fail) with error message:
```
Error while setting up credentials. If you are running a SQL warehouse cluster, you will need to defined stored credentials.
```
Setting up storage credentials is not something we can do in the Azure subscription as we lack permission. For more detail on what permissions we are lacking and how you could do it if you had the permission, see the [wiki article](https://analytics.dataiku.com/projects/DATAOPSWIKI/wiki/726/Databricks%20connector%20(v11.2)). An alternative is to use an ADLS connection with a SAS token.This will allow fast-write to work as intended, so change the fast-write connection to *common-adls-sas* in your SQL Databricks connection and rerun the Python recipe in step 2, which should now work.

As for step 3, the SAS key approach does not work for the *Databricks to Azure* engine but it does for the *Azure to Databricks* engine. So when recreating recipe setup 3, use the DSS engine for the first half and then the *Azure to Databricks* engine for the second half. Make sure the ADLS connection you sync the data to is *common-adls-sas*. When syncing to/from ADLS the fast-write connection is ignored and the ADLS connection where the data is going from/to is used, including for credentials.


## 4 - Train a model in Databricks & store as an MLflow model object

We can now also test importing an MLflow model into DSS that has been trained within Databricks. To do this we need to 

- Setup an MLflow experiment in Databricks, that saves its artifacts to a location we can fetch them from. In this example we'll mount an ADLS location to the DBFS, and then save artifacts there.
- Run an experiment, logging an MLflow model
- Reach this model from DSS. We'll create a managed folder, pointing to the ADLS location where the model will be saved
- Use the DSS Python API to import the model

### 4.1 - Pre-requisities if you skipped steps 1-3

Skip this if you created your own Databricks workspace connection, clusters, etc

We have pre-created a [Databricks workspace](https://portal.azure.com/#@dkudemos.onmicrosoft.com/resource/subscriptions/82852abb-55ae-44d8-9bac-4632a4173215/resourceGroups/databricks-demo/providers/Microsoft.Databricks/workspaces/databricks-demo-common/overview), and cluster located [here](https://adb-7661646899816156.16.azuredatabricks.net), let me (Mike) know if you need access and I'll add you to this workspace. There is a DS&E cluster called communal_cluster that you can use. 

Check you can login and see this cluster. If you're going to use this cluster, please name artifacts you create later personally to avoid confusion between users.

You'll also need create and record a secret for this [registered app](https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationMenuBlade/~/Credentials/appId/ed5f47b2-16d8-4f3b-abb2-cfe9337a5705/isMSAApp~/false). We'll refer to this later as `<YOUR_CLIENT_SECRET>`.

### 4.2 - Mount the Storage Account to a DBFS location

Inside the Databriks UI, go to the ML workspace (as opposed to SQL or DS&E), which shares the cluster with DS&E so there is no need to add compute. Inside this ML workspace, create a new notebook by clicking *New > Notebook > Python*. Inside the notebook run the following code to mount our ADLS container to the DBFS, replacing with a mount name (e.g. `mosborne_adls_mount`) and using your own client secret:
```
dbutils.fs.mount(
  source='abfss://databricks-demo@databricksdemodss.dfs.core.windows.net/',
  mount_point='/mnt/<MOUNT_NAME>',
  extra_configs={
    "fs.azure.account.auth.type": "OAuth",
    "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
    "fs.azure.account.oauth2.client.id": "ed5f47b2-16d8-4f3b-abb2-cfe9337a5705",
    "fs.azure.account.oauth2.client.secret": "<YOUR_CLIENT_SECRET>",
    "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/3ceb0d29-d7de-4204-b431-3f9f8edb2106/oauth2/token"
  }
)
```
There are [more secure ways](https://learn.microsoft.com/en-us/azure/databricks/security/secrets/) to reference secrets within Databricks code, but for speed/ease just hardcode it now.

### 4.3 - Create an MLflow experiment with artifact location on ADLS

Still within the ML workspace, create a new experiment by clicking *New > Experiment*. Name the experiment and provide the Artifact Location `dbfs:/mnt/<MOUNT_NAME>/<PERSONALISED_ADLS_PATH>`. For example `dbfs:/mnt/mosborne_adls_mount/mosborne/mlflow_experiments`. Once the experiment has been created, note down the experiment path by clicking the copy button next to experiment name at the top of the screen.

### 4.4 - Create a model within this experiment

Return back to your notebook (*Workspace > Users > <YOUR_USER> > <NOTEBOOK_NAME>*) or create a new one. Inside a new cell, paste the following and run without modification. It is sufficient code to fetch data and train a model inside a function, and requires not editing by you.

```
# Import required libraries
import os
import warnings
import sys

import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

# Import mlflow
import mlflow
import mlflow.sklearn

# Load diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Create pandas DataFrame 
Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
data = pd.DataFrame(d, columns=cols)

def train_diabetes(data, in_alpha, in_l1_ratio, experiment):
    
    mlflow.set_experiment(experiment)

    # Evaluate metrics
    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
    train_x = train.drop(["progression"], axis=1)
    test_x = test.drop(["progression"], axis=1)
    train_y = train[["progression"]]
    test_y = test[["progression"]]

    if float(in_alpha) is None:
        alpha = 0.05
    else:
        alpha = float(in_alpha)

    if float(in_l1_ratio) is None:
        l1_ratio = 0.05
    else:
        l1_ratio = float(in_l1_ratio)

    # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Print out ElasticNet model metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Log mlflow attributes for mlflow UI
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(lr, "model")
```
In a new cell run the following, altering `YOUR_EXPERIMENT_PATH` to align with what you copied in step 4.3
```
train_diabetes(data, 0.01, 0.01, "<YOUR_EXPERIMENT_PATH>")
```
If this runs without errors then it was successful. Check the [Storage account](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2F82852abb-55ae-44d8-9bac-4632a4173215%2Fresourcegroups%2Fdatabricks-demo%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fdatabricksdemodss/path/databricks-demo/etag/%220x8DB091F616CC6F7%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) and confirm your model artefact can be found. It should be located in the `databricks-demo` container and then `/<PERSONALISED_ADLS_PATH>/<UUID>/artifacts/model`.


### Create a managed folder in your project

To get DSS to access this location, we'll create a managed folder pointing to the same location, and then simply reference the managed folder when importing in the model.

Inside the same DSS project used previously, create a managed folder, using the *common-adls* connection. Edit the settings of the managed folder to point to your experiment's *artifacts* directory (the one containing *model*) and save. Copy the managed folder's ID from the URL and note down.

### Import your MLflow model

We now import the MLflow model using the DSS Python API. When importing the model, you need to associate it with a DSS code env, that aligns with the env used to train the model. This is a deliberately manual process. Using the managed folder, open up the model directory, and then the *requirements.txt* file. Copy the contents and use it to make a code environment, using Python 3.9 to avoid dependency issues since that is what Databricks used to train the model, and leaving core packages and Jupyter support.

Paste the requirements, removing the `<3,>=2.1` from the `mlflow` dependency, leaving only `mlflow` without version specification. This is because dataiku only supports up to [version 1.21.0](https://doc.dataiku.com/dss/latest/mlops/mlflow-models/limitations.html). Here we are relying on any differences between these two `mlflow` versions not causing issues when importing this model.

Once the packages have been added to the env, update it. Then in your project, open a Python notebook. Use the created code env. We're only importing the model here, so in theory we don't need to use the env we just created as the ML packages are superfluous, but we do need an env that has `mlflow` installed and our env has that so we can avoid creating a whole new code env but just reusing our model code env for the import.

Run the following lines inside the notebook. This will create a saved model object inside DSS called `imported_from_db` and then create a version of this model `0` that loads the MLflow model from your managed folder 
```
import dataiku
import mlflow

project = dataiku.api_client().get_default_project()
mlflow_model = project.create_mlflow_pyfunc_model('imported_from_db')

mlflow_model_version = mlflow_model.import_mlflow_version_from_managed_folder(
  version_id="0", 
  managed_folder=<YOUR_MANAGED_FOLDER_ID>, 
  path="model",
  code_env_name="<YOUR_CODE_ENV>"
)
```

### Check the model has been imported

Inside the same project, go to *Saved models* (inside the doughnut menu? icon) and confirm your model is present.