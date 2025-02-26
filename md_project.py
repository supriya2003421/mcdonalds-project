{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "cb69ccd3-ee53-47cb-894f-4aaed9b741b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "from sklearn.model_selection import train_test_split\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "3f1caa9e-d8f4-4660-a7f5-c8e7de23651f",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv(\"mcdonalds.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "679224bd-c410-44de-8daa-62f48cfb1deb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>yummy</th>\n",
       "      <th>convenient</th>\n",
       "      <th>spicy</th>\n",
       "      <th>fattening</th>\n",
       "      <th>greasy</th>\n",
       "      <th>fast</th>\n",
       "      <th>cheap</th>\n",
       "      <th>tasty</th>\n",
       "      <th>expensive</th>\n",
       "      <th>healthy</th>\n",
       "      <th>disgusting</th>\n",
       "      <th>Like</th>\n",
       "      <th>Age</th>\n",
       "      <th>VisitFrequency</th>\n",
       "      <th>Gender</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>-3</td>\n",
       "      <td>61</td>\n",
       "      <td>Every three months</td>\n",
       "      <td>Female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>+2</td>\n",
       "      <td>51</td>\n",
       "      <td>Every three months</td>\n",
       "      <td>Female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>+1</td>\n",
       "      <td>62</td>\n",
       "      <td>Every three months</td>\n",
       "      <td>Female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>+4</td>\n",
       "      <td>69</td>\n",
       "      <td>Once a week</td>\n",
       "      <td>Female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>+2</td>\n",
       "      <td>49</td>\n",
       "      <td>Once a month</td>\n",
       "      <td>Male</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  yummy convenient spicy fattening greasy fast cheap tasty expensive healthy  \\\n",
       "0    No        Yes    No       Yes     No  Yes   Yes    No       Yes      No   \n",
       "1   Yes        Yes    No       Yes    Yes  Yes   Yes   Yes       Yes      No   \n",
       "2    No        Yes   Yes       Yes    Yes  Yes    No   Yes       Yes     Yes   \n",
       "3   Yes        Yes    No       Yes    Yes  Yes   Yes   Yes        No      No   \n",
       "4    No        Yes    No       Yes    Yes  Yes   Yes    No        No     Yes   \n",
       "\n",
       "  disgusting Like  Age      VisitFrequency  Gender  \n",
       "0         No   -3   61  Every three months  Female  \n",
       "1         No   +2   51  Every three months  Female  \n",
       "2         No   +1   62  Every three months  Female  \n",
       "3        Yes   +4   69         Once a week  Female  \n",
       "4         No   +2   49        Once a month    Male  "
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "babdf1c1-1b29-4890-8c10-20368accfebd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1453 entries, 0 to 1452\n",
      "Data columns (total 15 columns):\n",
      " #   Column          Non-Null Count  Dtype \n",
      "---  ------          --------------  ----- \n",
      " 0   yummy           1453 non-null   object\n",
      " 1   convenient      1453 non-null   object\n",
      " 2   spicy           1453 non-null   object\n",
      " 3   fattening       1453 non-null   object\n",
      " 4   greasy          1453 non-null   object\n",
      " 5   fast            1453 non-null   object\n",
      " 6   cheap           1453 non-null   object\n",
      " 7   tasty           1453 non-null   object\n",
      " 8   expensive       1453 non-null   object\n",
      " 9   healthy         1453 non-null   object\n",
      " 10  disgusting      1453 non-null   object\n",
      " 11  Like            1453 non-null   object\n",
      " 12  Age             1453 non-null   int64 \n",
      " 13  VisitFrequency  1453 non-null   object\n",
      " 14  Gender          1453 non-null   object\n",
      "dtypes: int64(1), object(14)\n",
      "memory usage: 170.4+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "aef52346-392d-4aa5-ac68-273471b592bc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "yummy             0\n",
      "convenient        0\n",
      "spicy             0\n",
      "fattening         0\n",
      "greasy            0\n",
      "fast              0\n",
      "cheap             0\n",
      "tasty             0\n",
      "expensive         0\n",
      "healthy           0\n",
      "disgusting        0\n",
      "Like              0\n",
      "Age               0\n",
      "VisitFrequency    0\n",
      "Gender            0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(pd.isnull(data).sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "2d472bd1-2bc2-4bc3-a49c-3d12394fe629",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1453.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>44.604955</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>14.221178</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>18.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>33.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>45.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>57.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>71.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               Age\n",
       "count  1453.000000\n",
       "mean     44.604955\n",
       "std      14.221178\n",
       "min      18.000000\n",
       "25%      33.000000\n",
       "50%      45.000000\n",
       "75%      57.000000\n",
       "max      71.000000"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "1ac3e549-4d8e-41a7-8fc1-3905f019126d",
   "metadata": {},
   "outputs": [],
   "source": [
    "data[\"Gender\"] = data[\"Gender\"].map({\"Male\": 0, \"Female\": 1})\n",
    "visit_map = {\"Never\": 0, \"Once a year\": 1, \"Every three months\": 2, \"Once a month\": 3, \"Once a week\": 4, \"More than once a week\": 5}\n",
    "data[\"VisitFrequency\"] = data[\"VisitFrequency\"].map(visit_map)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "a33d83ec-11e6-46c6-b95d-3d3d300eebee",
   "metadata": {},
   "outputs": [],
   "source": [
    "perception_columns = [\"yummy\", \"convenient\", \"spicy\", \"fattening\", \"greasy\", \"fast\", \"cheap\", \"tasty\", \"expensive\", \"healthy\", \"disgusting\",\"Like\", \"Age\",\"VisitFrequency\", \"Gender\" ]\n",
    "data[perception_columns] = data[perception_columns].map(lambda x: 1 if x == \"Yes\" else 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "6b934a77-5209-4497-b4ac-24fd5aa370e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = data[[\"Age\", \"Gender\", \"VisitFrequency\", \"Like\"] + perception_columns]\n",
    "y = data[\"VisitFrequency\"]  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "id": "7cd04f24-bcec-4fd0-8cc0-49e50dc3d3b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "4af69d0a-46eb-4476-97f6-e03253363874",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "9bd842d3-a419-48ec-9814-56d9a1a307ce",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>KNeighborsClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">KNeighborsClassifier</label><div class=\"sk-toggleable__content\"><pre>KNeighborsClassifier()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "KNeighborsClassifier()"
      ]
     },
     "execution_count": 104,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "knn = KNeighborsClassifier(n_neighbors=5)\n",
    "knn.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "id": "c5d43ace-6938-4ff9-a78c-3174fec85e3b",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = knn.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "id": "c91b3f6b-2ebc-4be8-ae2d-e123d57e06a5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\supri\\anaconda3\\Lib\\site-packages\\seaborn\\matrix.py:260: FutureWarning: Format strings passed to MaskedConstant are ignored, but in future may error or produce different behavior\n",
      "  annotation = (\"{:\" + self.fmt + \"}\").format(val)\n"
     ]
    },
    {
     "data": {
      "image/png": "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hxdxq95ix5KIiIgoy5f6Dm+xfLr5XiIiIiL6rDBjSURERJSFGcvCYcfyK3Bbu4roMa2vnxM95tPKtUWPqX85TPSYdkaJose8J3cVPaaQLv5f7smGRqLHdDGMFzWeBEXziJG8tPR8IXpMwKwYYhIRh8KJiIiISC2YsSQiIiLK8ik/buhzwLtHRERERGrBjCURERFRFi7eKRxmLImIiIhILZixJCIiIsrCOZaFw7tHRERERGrBjCURERFRNgnnWBYGM5ZEREREpBbMWBIRERFl4arwwmHGUmTe3t4YOXJkcTeDiIiISO2YsRTZzp07oa2tXdzNICIiIhW4Krxw2LEUmZmZWXE3gYiIiKhIsFv+EXbs2IEqVapAT08P5ubmaNq0KVJTU+Hv74/27dtj+vTpsLKygpGREQYPHoy0tDTFse8PhctkMowfPx52dnaQSqVwcnLCmjVrIAgCypcvj7lz5yrFvn79OjQ0NBAZGSnW5RIREX01JBoS0bYvETOWBRQTE4MePXpg9uzZ6NChA5KTk3Hy5EkIggAAOHLkCHR1dXH06FFERUWhb9++sLCwwKxZs1Ser3fv3jh79iwWL14Md3d33Lt3D3FxcZBIJOjXrx+CgoIwduxYRf21a9eifv36KFeunCjXS0RERJRf7FgWUExMDDIyMtCxY0eUKVMGAFClShXFfh0dHaxduxYlSpRApUqVMGPGDIwbNw4zZ86ExnvzNm7fvo3t27cjJCQETZs2BQCULVtWsb9v376YMmUKLly4gJo1ayI9PR0bN27EnDlzRLhSIiKirw/nWBYO714Bubu7o0mTJqhSpQq6dOmCVatWITExUWl/iRIlFD97eXkhJSUFDx48yHGusLAwaGpqomHDhipj2draolWrVli7di0AYN++fXj9+jW6dOmSa/tkMhlevHihtKWlyT72comIiIjyjR3LAtLU1ERISAj+/vtvVKxYEUuWLIGzszPu3buX53ESFU/y19PT+2C8AQMGYOvWrXj16hWCgoLQrVs3pY7r+wIDA2FsbKy0rf99/ocvjIiIiDjHspDYsfwIEokEdevWxfTp03H58mXo6Ohg165dAIArV67g1atXirrnzp2DgYEBSpcuneM8VapUgVwux/Hjx3ON1bJlS+jr62P58uX4+++/0a9fvzzbFhAQgKSkJKWt96DRH3mlRERERPnHOZYF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",
      "text/plain": [
       "<Figure size 700x700 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(7,7))\n",
    "sns.heatmap(data.corr(), annot=True, cmap=\"coolwarm\", fmt=\".2f\")\n",
    "plt.title(\"Feature Correlation Heatmap\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "id": "785bbe9c-12de-48ab-ba2f-b1db0f5725ca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Accuracy: 1.00\n",
      "\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00       291\n",
      "\n",
      "    accuracy                           1.00       291\n",
      "   macro avg       1.00      1.00      1.00       291\n",
      "weighted avg       1.00      1.00      1.00       291\n",
      "\n"
     ]
    }
   ],
   "source": [
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(f\"Model Accuracy: {accuracy:.2f}\")\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8656e6ec-27f8-44e9-a5d5-aca903c1b8ad",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
