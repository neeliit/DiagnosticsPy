import numpy as np

class regressor():
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
        #................Mean absolute error...............#
        from sklearn.metrics import mean_absolute_error
        MAE = round(mean_absolute_error(y_test,y_pred),2)

        #..................Mean squared error...............#
        from sklearn.metrics import mean_squared_error
        MSE = round(mean_squared_error(y_test,y_pred),2)

        #.................Root mean squared error....................#
        RMSE = round(np.sqrt(mean_squared_error(y_test,y_pred)),2)

        #................Root mean squared log error................#
        RMSELog = round(np.log(np.sqrt(mean_squared_error(y_test,y_pred))),2)

        #.....................R-Square .....................#
        from sklearn.metrics import r2_score
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
        errorList = np.array([MAE, MSE, RMSE, RMSELog, r2, Adj_R2])
        # errorsName = ['MAE','MSE','RMSE','RMSELog','R-sqr','Adj_R2']
        return(errorList)
    
    def rf(self, x_train, x_test, y_train, y_test):
        #modname = {'Model name':'Random Forest',
                   #'Model function':'rf'}
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(n_estimators=100,random_state=0)
        reg.fit(x_train,y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)
        
        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
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
    
    def lasso(self, x_train, x_test, y_train, y_test):
        from sklearn.linear_model import Lasso
        reg = Lasso(alpha=0.1)
        reg.fit(x_train, y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)
        
        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    def ridge(self, x_train, x_test, y_train, y_test):
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=0.1)
        reg.fit(x_train, y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)

        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    def en(self, x_train, x_test, y_train, y_test):
        from sklearn.linear_model import ElasticNet
        reg = ElasticNet(random_state=0)
        reg.fit(x_train, y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)
        
        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    def llar(self, x_train, x_test, y_train, y_test):
        #..........Training the Simple Linear Regression model on Training set.....#
        from sklearn.linear_model import Lars
        reg = Lars()  
        reg.fit(x_train,y_train)  

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)

        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    def huber(self, x_train, x_test, y_train, y_test):
        #..........Training the Simple Linear Regression model on Training set.....#
        from sklearn.linear_model import HuberRegressor
        reg = HuberRegressor()  
        reg.fit(x_train,y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)

        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    def br(self, x_train, x_test, y_train, y_test):
        #..........Training the Simple Linear Regression model on Training set.....#
        from sklearn.linear_model import BayesianRidge
        reg = BayesianRidge() 
        reg.fit(x_train,y_train)  

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)

        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    def gbr(self, x_train, x_test, y_train, y_test):
        from sklearn.ensemble import GradientBoostingRegressor
        reg = GradientBoostingRegressor(random_state=0)
        reg.fit(x_train, y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)

        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    def etr(self, x_train, x_test, y_train, y_test):
        from sklearn.ensemble import ExtraTreesRegressor
        reg = ExtraTreesRegressor(n_estimators=100, random_state=0)
        reg.fit(x_train,y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)

        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    def xgb(self, x_train, x_test, y_train, y_test):
        from xgboost import XGBRegressor
        reg = XGBRegressor(objective='reg:squarederror')
        reg.fit(x_train, y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)
        
        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    def knn(self, x_train, x_test, y_train, y_test):
        from sklearn.neighbors import KNeighborsRegressor
        reg = KNeighborsRegressor(n_neighbors=2)
        reg.fit(x_train, y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)

        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    def ctb(self, x_train, x_test, y_train, y_test):
        from catboost import CatBoostRegressor
        reg = CatBoostRegressor(verbose=0, n_estimators=100)
        reg.fit(x_train, y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)

        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    def lgbmr(self, x_train, x_test, y_train, y_test):
        from lightgbm import LGBMRegressor
        reg = LGBMRegressor(verbose=0, n_estimators=100)
        reg.fit(x_train, y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)

        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    def dummy(self, x_train, x_test, y_train, y_test):
        from sklearn.dummy import DummyRegressor
        reg = DummyRegressor(strategy="mean")
        reg.fit(x_train, y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)

        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    def par(self, x_train, x_test, y_train, y_test):
        from sklearn.linear_model import PassiveAggressiveRegressor
        reg = PassiveAggressiveRegressor(max_iter=100, random_state=0)
        reg.fit(x_train, y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)
        
        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)
    
    def dt(self, x_train, x_test, y_train, y_test):
        from sklearn.tree import DecisionTreeRegressor
        reg = DecisionTreeRegressor(random_state=0)
        reg.fit(x_train,y_train)

        #.......Predict the Test set result.......#
        y_pred = reg.predict(x_test)

        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore)       
    
    def svr(self, x_train, x_test, y_train, y_test):
        #.............Feature Scaling............#
        from sklearn.preprocessing import StandardScaler
        sc_x = StandardScaler()
        sc_y = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        y_train = sc_y.fit_transform(y_train)

        #..........Training the SVR on training set.....#
        from sklearn.svm import SVR
        reg = SVR(kernel='rbf') # radial basis function kernal
        reg.fit(x_train,y_train)

        #.......Predict the Test set result.......#
        y_pred = sc_y.inverse_transform(reg.predict(sc_x.transform(x_test)).reshape(-1,1))
        
        #............Cross validation............#
        cvscore = self.crossval(reg.fit(x_train,y_train),x_train, y_train)

        #.................Score..................#
        scores = self.scores(y_test, y_pred, x_train)
        return(y_pred,scores,cvscore) 


        
        
        
    


    

    

    

    