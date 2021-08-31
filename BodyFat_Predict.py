

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def normalize(X): 
    
    mins = np.min(X, axis = 0) 
    maxs = np.max(X, axis = 0) 
    rng =( maxs - mins) 
    norm_X = (X-mins)/rng 
    
    return norm_X,maxs,rng

def sigmoid_func(beta, X):
    
    return 1.0/(1 + np.exp(-np.dot(X, beta.T))) 

def log_gradient(beta, X, y): 
    
    first_calc = sigmoid_func(beta, X) - y
    final_calc = np.dot(first_calc.T, X) 
    return final_calc 

def cost_func(beta, X, y): 
    
    log_func_v = sigmoid_func(beta, X) 
    y = np.squeeze(y) 
    step1 = y * np.log(log_func_v) 
    step2 = (1 - y) * np.log(1 - log_func_v) 
    final = -step1 - step2 
    return np.mean(final) 

def grad_descent(X, y, beta, lr=.01, converge_change=.001): 
    
    cost = cost_func(beta, X, y) 
    change_cost = 1
    num_iter = 1
      
    while(change_cost > converge_change): 
        old_cost = cost 
        beta = beta - (lr * log_gradient(beta, X, y)) 
        cost = cost_func(beta, X, y) 
        change_cost = old_cost - cost 
        num_iter += 1
      
    return beta, num_iter  

def predict(beta, X): 
    pred_prob = sigmoid_func(beta, X) 
    pred_value = np.where(pred_prob >= 0.5, 1, 0) 
    return np.squeeze(pred_value) 


if __name__ == "__main__": 

    data =pd.read_csv(r"Bodyfat-Levels.csv")
    X= data.drop(['level','bodyfatlevel'] , axis=1) 
    y=data['level']
    # Randomly select 10% of the dataset records and save them into Bodyfat-Level-Test-Data.csv file
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=4)
    test_data=pd.DataFrame(x_test)
    test_data.insert(14,"Level",y_test)
    test_data.to_csv("TestBodyfat-Levels.csv", header = True, index = True)

    # Classifying the classes
    y_class1 = [1 if each == "Essential fat" else 0 for each in y_train]
    y_class2 = [1 if each == "Athletes" else 0 for each in y_train]
    y_class3 = [1 if each == "Fitness" else 0 for each in y_train]
    y_class4 = [1 if each == "Acceptable" else 0 for each in y_train]
    y_class5 = [1 if each == "Obesity" else 0 for each in y_train]
    
    
    m = len(y_class1)
    x_train=(x_train - np.min(x_train))/(np.max(x_train) - np.min(x_train)).values 
    x_train = np.hstack((np.ones((m,1)), x_train))
    theta0= np.zeros(15) 
    
    # Essential fat
    theta1, num_iter1 = grad_descent(x_train, y_class1, theta0) 
    print("Estimated regression coefficients OF Essential fat:", theta1) 
    print("No. of iterations OF Essential fat:", num_iter1,"\n")
    
    # Athletes
    theta2, num_iter2 = grad_descent(x_train, y_class2, theta0) 
    print("Estimated regression coefficients OF Athletes:", theta2) 
    print("No. of iterations OF Athletes:", num_iter2,"\n") 
    
    # Fitness 
    theta3, num_iter3 = grad_descent(x_train, y_class3, theta0) 
    print("Estimated regression coefficients OF Fitness:", theta3) 
    print("No. of iterations OF Fitness:", num_iter3,"\n") 
    
    # Acceptable
    theta4, num_iter4= grad_descent(x_train, y_class4, theta0) 
    print("Estimated regression coefficients OF Acceptable:", theta4) 
    print("No. of iterations OF Acceptable:", num_iter4,"\n")
    
    # Obesity
    theta5, num_iter5 = grad_descent(x_train, y_class5, theta0)  
    print("Estimated regression coefficients OF Obesity:", theta5) 
    print("No. of iterations OF Obesity:", num_iter5,"\n") 
    
    # predicted labels 
    y_pred1 = predict(theta1, x_train) 
    print ("y_pred of Essential fat= ", y_pred1) 
    y_pred2 = predict(theta2, x_train) 
    print ("y_pred of Athletes= ", y_pred2) 
    y_pred3 = predict(theta3, x_train) 
    print ("y_pred of Fitness= ", y_pred3)
    y_pred4 = predict(theta4, x_train) 
    print ("y_pred of Acceptable= ", y_pred4)
    y_pred5 = predict(theta5, x_train) 
    print ("y_pred of Obesity= ", y_pred5)
    
    #final max between 5 classes of the train test
    Final_y_pred=[]
    for i in range(len(y_train)):
        ms= max(y_pred1[i],y_pred2[i],y_pred3[i],y_pred4[i],y_pred5[i])
        if ms==y_pred1[i] :
            Final_y_pred.append('Essential fat')
        elif ms==y_pred2[i]:
            Final_y_pred.append('Athletes')
        elif ms==y_pred3[i]:
            Final_y_pred.append('Fitness')
        elif ms==y_pred4[i]:
            Final_y_pred.append('Acceptable')
        elif ms==y_pred5[i]:
            Final_y_pred.append('Obesity')
            
    print("list of predict of train data: ",Final_y_pred )
    print("\n")
    print("Accuracy of final prediction of y: ",accuracy_score(y_train,Final_y_pred)*100,"%")
    
#Using Test data for prediction

    Test_data=pd.read_csv(r'TestBodyfat-Levels.csv')
    x_test=Test_data.iloc[:, 1:15]
    y_test=Test_data.iloc[:, 15]
    m = len(y_test)
    x_test,maxs,rng = normalize(x_test)  
    x_test = np.hstack((np.ones((m,1)), x_test)) 
    
#Test Data prediction for 5 classes
    y_predtest1 = predict(theta1, x_test) 
    
    y_predtest2 = predict(theta2, x_test) 
    
    y_predtest3 = predict(theta3, x_test) 
    
    y_predtest4 = predict(theta4, x_test) 
    
    y_predtest5 = predict(theta5, x_test) 
      
    Final_y_predtest=[]
    for i in range(len(y_test)):
        Max= max(y_predtest1[i],y_predtest2[i],y_predtest3[i],y_predtest4[i],y_predtest5[i])
        if Max==y_predtest1[i] :
            Final_y_predtest.append('Essential fat')
        elif Max==y_predtest2[i]:
            Final_y_predtest.append('Athletes')
        elif Max==y_predtest3[i]:
            Final_y_predtest.append('Fitness')
        elif Max==y_predtest4[i]:
            Final_y_predtest.append('Acceptable')
        elif Max==y_predtest5[i]:
            Final_y_predtest.append('Obesity')
    print("list of predict of test data: ")
    print (Final_y_predtest)
    print("Final accuracy of test data ",accuracy_score(y_test,Final_y_predtest)*100,"%")

