import numpy as np

class classifiers ():
    def __init__(self,x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def crossval(self,modelcv,x_train, y_train):
        from sklearn.model_selection import KFold, cross_val_score
        cv = KFold(n_splits=6, random_state=0, shuffle=True)

        score = cross_val_score(modelcv, X=x_train, y=y_train, scoring='r2',
                         cv=cv, n_jobs=-1)
        
        cross_val_list = np.array([score.mean(), score.std()])
        return(cross_val_list)
    
    def scores(self, y_test, y_pred, x_train):
        #y_pred = self.fit(x_train, x_test, y_train)
        from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
        cm = confusion_matrix(y_test,y_pred)
        total=sum(sum(cm))
        
        #.....................Accuracy......................#
        accuracy = accuracy_score(y_test,y_pred)
        
       #.....................Senstivity.....................#
        sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

        #.....................Specificity...................#
        specificity = cm[1,1]/(cm[1,0]+cm[1,1])
        
        #.....................R-Square .....................#
        r2 = round(r2_score(y_test,y_pred),2)

        #..................Adjusted R-sqaure.................#
        # number of observations
        # number of independent variables
        # r2 is r sqaure calculated previously
        n = x_train.shape[0]
        k = x_train.shape[1]
        Adj_R2 = round(1 - ((1-r2)*(n-1)/(n-k-1)),2)

        # scoresdict = {
        #     'MAE': MAE,
        #     'MSE': MSE,
        #     'RMSE': RMSE,
        #     'RMSELog': RMSELog,
        #     'R-sqr':r2,
        #     'Adj_R2': Adj_R2
        # }
        errorList = np.array([accuracy, sensitivity, specificity, r2, Adj_R2])
        # errorsName = ['MAE','MSE','RMSE','RMSELog','R-sqr','Adj_R2']
        return(errorList)
    #......................Logistic regression.....................#
    def logreg(self, x_train, x_test, y_train, y_test):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=0)
        classifier.fit(x_train,y_train)

        #.......Predict the Test set result.......#
        y_pred = classifier.predict(x_test)
        #y_pred_quant = classifier.predict_proba(x_test)[:, 1] #Only keep the first column, which is the 'pos' values

    #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores)
        # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_quant)
        # metrics.auc(fpr, tpr)

    #.......................Decision tree .........................#
    def dt(self, x_train, x_test, y_train, y_test):    
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion= 'entropy',random_state=0)
        classifier.fit(x_train,y_train)

        #.......Predict the Test set result.......#
        y_pred = classifier.predict(x_test)
        #y_pred_quant = classifier.predict_proba(x_test)[:, 1] #Only keep the first column, which is the 'pos' values

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores)
    
    
        
    def lr(self, x_train, x_test, y_train, y_test):
        #..........Training the Simple Linear Regression model on Training set.....#
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression(fit_intercept=True)  
        reg.fit(x_train,y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)

        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    
        
        
        
    


    

    

    

    