import pandas as pd
import math
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#### Feature Selection #####

# calculating the entropy (no use here)
def entropy(col):
    count = Counter(col)
    unique_dict = {}
    
    for key, value in count.items():
        unique_dict[key] = value
        
    print(unique_dict)
    print("\n");
    val = 0
    
    for i in unique_dict.keys():
        x = unique_dict.get(i)
        val += (x/len(col))*math.log2(x/len(col))
    
    return val*(-1);


# it is to create a dicitonary have key as each unique value in array and value of the key is the count of that unique value
def dictCreator(col):
    count = Counter(col)
    unique_dict = {}
    
    for key, value in count.items():
        unique_dict[key] = value
        
    return unique_dict;


if __name__ == '__main__':
    df = pd.read_csv(r'D:\Source_Codes\DM_lab\Iris.csv');
    print(df.shape);
    #inputs so df -> x and df ->label
    label = df['Species']
    df.drop("Species", axis=1, inplace=True)
    df.drop("Id", axis=1, inplace=True)
    print(df.shape);
    print(label.value_counts())
    features = set(df.columns) #has feeature names
    print(features)
    scaler = MinMaxScaler() # this is to normalize (doing the min max normalization)
    df_norm = df.copy() #copying the data to df_norm
    df_norm[list(features)] = scaler.fit_transform(df[list(features)])
    print(df_norm)
    
    print(df)
    
    featuresList = list(features);
    columnDict = {};
    
    #joint Count for the calculating the joint count i.e where feature[i] and Species occur happen simultaneously
    jointCount = [];
    #single Count is the count of the only either for the feature[i] or the Species
    singleCount = [];
    
    labelSet = set(label)
    print(labelSet)
    
    print(featuresList)
    
    
    #creating the array of dictionary of dicitionary (array of columns , 1st key as the Species name , and inside level 2nd Keys - the unique values in that attribute for the 1st Key)
    for i in range (0,len(featuresList)):
        
        tempDict = dictCreator(df_norm[featuresList[i]]);
        singleCount.append(tempDict)
        
        
        dictLabel = {}
        for j in labelSet:
            temp = []
            column = df_norm[featuresList[i]]
            for k in range (0,len(label)):
                if(label[k] == j):
                    temp.append(column[k])
            
            dictUnique = dictCreator(temp)  
            dictLabel[j] = dictUnique    
            
        jointCount.append(dictLabel)
        
    
     
    #single count is for the total count of value in whole column   
    singleCount.append(dictCreator(label));
    
        
    # for i in range(0, len(jointCount)):
    #     print(featuresList[i] + " : \n")
    #     print(jointCount[i])
    #     print(singleCount[i])
    #     print("\n")
        
    # print(singleCount[len(singleCount)-1])
    
    
    #calculating the mutual information 
    mutualInfo = [];
    
    for i in range(0, len(jointCount)):
        sum = 0;
        
        for j in jointCount[i].keys():
            x = jointCount[i].get(j)
            div1 = ((singleCount[len(singleCount)-1])[j])/len(df_norm) #getting the p(x)
            
            for k in x.keys():
                val = x.get(k)/len(df_norm) #getting the p(x,y)
                div2 = ((singleCount[i])[k])/len(df_norm) #getting the p(y)
                div = div1 * div2 #finding the p(x)*p(y)
                #X - Species i.e label unique values
                #Y - the Features unique values
                # FORMULA :- sumX { sumY { p(x,y).log2(p(x,y)/p(x).p(y)) } }
                sum += val*(math.log2(val/div))
                
        mutualInfo.append(sum)
       
       
    #Printing the mutual information for each feature :-
    print("the mutual information is :-")
    
    j = 0;
    for i in featuresList:
        print(i , " : " , mutualInfo[j]);
        j+=1;
        
    print("\n")
    
    
    x = int(input("Please enter the threshold: "));
    
    for i in range(0,len(featuresList)):
        if(mutualInfo[i] < x):
            df.drop(featuresList[i], axis=1, inplace=True)
            df_norm.drop(featuresList[i], axis=1, inplace=True)
            
            
    print(df);
    
    X = df
    y = label
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print(X_train)
    print("\n")
    print(X_test)
    print("\n")
    print(y_train)
    print("\n")
    print(y_test)
    print("\n")
    
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    print(y_pred)
    
    accuracy = accuracy_score(y_test, y_pred)   
    print("Accuracy:", accuracy)
    
    #feature selection ended
                
    
    
    
