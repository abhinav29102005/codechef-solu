import pandas as pd
import numpy as np
from scipy import stats

class EmployeePerformanceAnalysis:
    """
    Project: Clean and Explore Employee Dataset
    Statistical analysis, EDA, hypothesis testing, and linear regression on employee performance.
    """
    def __init__(self):
        self.dataset = pd.read_csv('employee_performance.csv')

    def perform_t_test(self):
        a=self.dataset.loc[self.dataset['Department']=='Engineering','PerformanceScore']
b=self.dataset.loc[self.dataset['Department']=='Sales','PerformanceScore']
s,p=stats.ttest_ind(a,b)
return {'t_statistic':float(s),'p_value':float(p)}

    def perform_chi_square_test(self):
        table=pd.crosstab(self.dataset['Department'],self.dataset['PromotionStatus'])
s,p,_,_=stats.chi2_contingency(table)
return {'chi_square_statistic':float(s),'p_value':float(p)}

    def calculate_correlation(self):
        r,p=stats.pearsonr(self.dataset['YearsOfExperience'],self.dataset['PerformanceScore'])
return {'correlation_coefficient':float(r),'p_value':float(p)}

    def calculate_confidence_interval(self):
        scores=self.dataset['PerformanceScore']
a,b=stats.t.interval(0.95,df=len(scores)-1,loc=scores.mean(),scale=stats.sem(scores))
return {'lower_bound':float(a),'upper_bound':float(b)}

    def perform_linear_regression(self):
        fit=stats.linregress(self.dataset['YearsOfExperience'],self.dataset['PerformanceScore'])
return {'slope':float(fit.slope),'intercept':float(fit.intercept),'correlation_coefficient':float(fit.rvalue)}

    def generate_statistical_report(self):
        return {'t_test':self.perform_t_test(),'chi_square_test':self.perform_chi_square_test(),'correlation':self.calculate_correlation(),'confidence_interval':self.calculate_confidence_interval(),'linear_regression':self.perform_linear_regression()}
