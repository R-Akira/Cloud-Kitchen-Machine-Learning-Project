#!/usr/bin/env python
# coding: utf-8

# In[3]:


import timeit


code_to_test = """

# Student Name : Reyner Akira
# Cohort       : 2

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################



import pandas as pd # data science essentials
from sklearn.model_selection import train_test_split # train/test split
import sklearn.linear_model  # linear regression (scikit-learn)
from sklearn.ensemble import GradientBoostingRegressor # gradient booster (scikit-learn)
import statsmodels.formula.api as smf # linear regression (statsmodels)





################################################################################
# Load Data
################################################################################


file = 'Apprentice_Chef_Dataset.xlsx'

original_df = pd.read_excel(file)



################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# Price Feature

original_df['price'] = original_df['REVENUE'] / original_df['TOTAL_MEALS_ORDERED']

# Expected price of meal price for making drink ordered feature

expected_price = (10+23)/2

# Drink Ordered Feature

original_df['expected_drink'] = original_df['price'] - expected_price 

drink_ordered = []
for i in original_df['expected_drink']:
    if i > 0:
        temp = i
        drink_ordered.append(temp)
    else:
        temp = 0
        drink_ordered.append(temp) 
        
    

original_df['drink_ordered'] = drink_ordered

###### Email Domain Engineering Part 1 ###### 

# placeholder list
placeholder_lst = []

# looping over each email address
for index, col in original_df.iterrows():
    
    # splitting email domain at '@'
    split_email = original_df.loc[index, 'EMAIL'].split(sep = '@')
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)
    

# converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)


# displaying the results
email_df.columns = ['name','email']


# concatenating personal_email_domain with friends DataFrame
original_df = pd.concat([original_df, email_df],
                   axis = 1)

###### Email Domain Engineering Part 2 ###### 


personal_email = ['@gmail.com','@protonmail.com','@yahoo.com','@msn.com',
                 '@aol.com','@passport.com','@hotmail.com','@live.com','@me.com']


work_email = ['@amex.com',"@cocacola.com",'@jnj.com','@mcdonalds.com',
             '@merck.com','@nike.com','@apple.com','@ge.org','@dupont.com',
             '@ibm.com','@chevron.com','@microsoft.com','@unitedhealth.com',
             '@exxon.com','@travelers.com','@boeing.com','@pg.com','@caterpillar.com',
             '@verizon.com','@mmm.com','@walmart.com','@disney.com','@pfizer.com',
             '@visa.com','@jpmorgan.com','@cisco.com','@goldmansacs.com',
             '@unitedtech.com','@intel.com','@homedepot.com']

# placeholder list
placeholder_lst = []


# looping to group observations by domain type
for domain in original_df['email']:
        if "@" + domain in personal_email:
            placeholder_lst.append('personal')
            
        elif "@"+ domain in work_email:
            placeholder_lst.append('work')
            
        else:
            print('Unknown')
            
# concatenating with original DataFrame
original_df['domain_group'] = pd.Series(placeholder_lst)

###### Email Domain Engineering Part 3 ###### 

# placeholder list
placeholder_lst = []

# looping to group observations by domain type
for i in original_df['domain_group']:
        if i == "work" :
            placeholder_lst.append(1)
            
        elif i == "personal":
            placeholder_lst.append(0)
            
        else:
            print('Unknown')
            
# concatenating with original DataFrame
original_df['domain_group_numeric'] = pd.Series(placeholder_lst)


##############################################################################
## Feature Engineering (Outlier Thresholds)                                 ##
##############################################################################

out_revenue_hi = 2300
out_total_meals_ordered_hi = original_df['TOTAL_MEALS_ORDERED'].quantile(0.95)
out_unique_meals_purch_hi = original_df['UNIQUE_MEALS_PURCH'].quantile(0.95)
out_contacts_w_customer_service_lo = original_df['CONTACTS_W_CUSTOMER_SERVICE'].quantile(0.025)
out_contacts_w_customer_service_hi = original_df['CONTACTS_W_CUSTOMER_SERVICE'].quantile(0.975)
out_avg_time_per_site_visit_hi = original_df['AVG_TIME_PER_SITE_VISIT'].quantile(0.95)
out_cancellations_before_noon_hi = original_df['CANCELLATIONS_BEFORE_NOON'].quantile(0.95)
out_cancellations_after_noon_hi = original_df['CANCELLATIONS_AFTER_NOON'].quantile(0.95)
out_mobile_logins_lo = original_df['MOBILE_LOGINS'].quantile(0.025)
out_mobile_logins_hi = original_df['MOBILE_LOGINS'].quantile(0.975)
out_pc_logins_lo = original_df['PC_LOGINS'].quantile(0.025)
out_pc_logins_hi = original_df['PC_LOGINS'].quantile(0.975)
out_weekly_plan_hi = original_df['WEEKLY_PLAN'].quantile(0.95)
out_early_deliveries_hi = original_df['EARLY_DELIVERIES'].quantile(0.95)
out_late_deliveries_hi =original_df['LATE_DELIVERIES'].quantile(0.95)
out_refrigerated_locker_hi = original_df['REFRIGERATED_LOCKER'].quantile(0.95)
out_followed_recommendations_pct_lo = original_df['FOLLOWED_RECOMMENDATIONS_PCT'].quantile(0.025)
out_followed_recommendations_pct_hi = original_df['FOLLOWED_RECOMMENDATIONS_PCT'].quantile(0.975)
out_avg_prep_vid_time_hi = original_df['AVG_PREP_VID_TIME'].quantile(0.95)
out_largest_order_size_lo = original_df['LARGEST_ORDER_SIZE'].quantile(0.05)
out_largest_order_size_hi = original_df['LARGEST_ORDER_SIZE'].quantile(0.975)
out_median_meal_rating_lo = original_df['MEDIAN_MEAL_RATING'].quantile(0.025)
out_median_meal_rating_hi = original_df['MEDIAN_MEAL_RATING'].quantile(0.975)
out_avg_clicks_per_visit_lo = original_df['AVG_CLICKS_PER_VISIT'].quantile(0.025)
out_avg_clicks_per_visit_hi = original_df['AVG_CLICKS_PER_VISIT'].quantile(0.975)
out_total_photos_viewed_lo = original_df['TOTAL_PHOTOS_VIEWED'].quantile(0.025)
out_total_photos_viewed_hi = original_df['TOTAL_PHOTOS_VIEWED'].quantile(0.975)
out_drink_ordered_hi = 30
out_price_hi = 60

##### Developing Features (Columns) for Outliers #####

# Revenue
original_df['out_revenue'] = 0
condition_hi = original_df.loc[0:,'out_revenue'][original_df['REVENUE'] > out_revenue_hi]

original_df['out_revenue'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Total Meals Ordered
original_df['out_total_meals_ordered'] = 0
condition_hi = original_df.loc[0:,'out_total_meals_ordered'][original_df['TOTAL_MEALS_ORDERED'] > out_total_meals_ordered_hi]

original_df['out_total_meals_ordered'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Unique Meals Purchased 
original_df['out_unique_meals_purch'] = 0
condition_hi = original_df.loc[0:,'out_unique_meals_purch'][original_df['UNIQUE_MEALS_PURCH'] > out_unique_meals_purch_hi]

original_df['out_unique_meals_purch'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Contacts With Customer Service
original_df['out_unique_meals_purch'] = 0
condition_lo = original_df.loc[0:,'out_unique_meals_purch'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] < out_contacts_w_customer_service_lo]
condition_hi = original_df.loc[0:,'out_unique_meals_purch'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > out_contacts_w_customer_service_hi]

original_df['out_unique_meals_purch'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

original_df['out_unique_meals_purch'].replace(to_replace = condition_lo,
                                value      = 1,
                                inplace    = True)

# Average Time per Site Visit
original_df['out_avg_time_per_site_visit'] = 0
condition_hi = original_df.loc[0:,'out_avg_time_per_site_visit'][original_df['AVG_TIME_PER_SITE_VISIT'] > out_avg_time_per_site_visit_hi]

original_df['out_avg_time_per_site_visit'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Cancellations Before Noon
original_df['out_cancellations_before_noon'] = 0
condition_hi = original_df.loc[0:,'out_cancellations_before_noon'][original_df['CANCELLATIONS_BEFORE_NOON'] > out_cancellations_before_noon_hi]

original_df['out_cancellations_before_noon'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Cancellations After Noon
original_df['out_cancellations_after_noon'] = 0
condition_hi = original_df.loc[0:,'out_cancellations_after_noon'][original_df['CANCELLATIONS_AFTER_NOON'] > out_cancellations_after_noon_hi]

original_df['out_cancellations_before_noon'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Mobile Logins
original_df['out_mobile_logins'] = 0
condition_lo = original_df.loc[0:,'out_mobile_logins'][original_df['MOBILE_LOGINS'] < out_mobile_logins_lo]
condition_hi = original_df.loc[0:,'out_mobile_logins'][original_df['MOBILE_LOGINS'] > out_mobile_logins_hi]

original_df['out_mobile_logins'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

original_df['out_mobile_logins'].replace(to_replace = condition_lo,
                                value      = 1,
                                inplace    = True)

# PC Logins
original_df['out_pc_logins'] = 0
condition_lo = original_df.loc[0:,'out_pc_logins'][original_df['PC_LOGINS'] < out_pc_logins_lo]
condition_hi = original_df.loc[0:,'out_pc_logins'][original_df['PC_LOGINS'] > out_pc_logins_hi]

original_df['out_pc_logins'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

original_df['out_pc_logins'].replace(to_replace = condition_lo,
                                value      = 1,
                                inplace    = True)

# Weekly Plan
original_df['out_weekly_plan'] = 0
condition_hi = original_df.loc[0:,'out_weekly_plan'][original_df['WEEKLY_PLAN'] > out_weekly_plan_hi]

original_df['out_weekly_plan'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Early Deliveries
original_df['out_early_deliveries'] = 0
condition_hi = original_df.loc[0:,'out_early_deliveries'][original_df['EARLY_DELIVERIES'] > out_early_deliveries_hi]

original_df['out_early_deliveries'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Late Deliveries
original_df['out_late_deliveries'] = 0
condition_hi = original_df.loc[0:,'out_late_deliveries'][original_df['LATE_DELIVERIES'] > out_late_deliveries_hi]

original_df['out_late_deliveries'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Refrigerated Locker 
original_df['out_refrigerated_locker'] = 0
condition_hi = original_df.loc[0:,'out_refrigerated_locker'][original_df['REFRIGERATED_LOCKER'] > out_refrigerated_locker_hi]

original_df['out_refrigerated_locker'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Followed Recommendations PCT
original_df['out_followed_recommendations_pct'] = 0
condition_lo = original_df.loc[0:,'out_followed_recommendations_pct'][original_df['FOLLOWED_RECOMMENDATIONS_PCT'] < out_followed_recommendations_pct_lo]
condition_hi = original_df.loc[0:,'out_followed_recommendations_pct'][original_df['FOLLOWED_RECOMMENDATIONS_PCT'] > out_followed_recommendations_pct_hi]

original_df['out_followed_recommendations_pct'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

original_df['out_followed_recommendations_pct'].replace(to_replace = condition_lo,
                                value      = 1,
                                inplace    = True)

# Average Prep Time Vid  
original_df['out_avg_prep_vid_time'] = 0
condition_hi = original_df.loc[0:,'out_avg_prep_vid_time'][original_df['AVG_PREP_VID_TIME'] > out_avg_prep_vid_time_hi]

original_df['out_avg_prep_vid_time'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Largest Order Size
original_df['out_largest_order_size'] = 0
condition_lo = original_df.loc[0:,'out_largest_order_size'][original_df['LARGEST_ORDER_SIZE'] < out_largest_order_size_lo]
condition_hi = original_df.loc[0:,'out_largest_order_size'][original_df['LARGEST_ORDER_SIZE'] > out_largest_order_size_hi]

original_df['out_largest_order_size'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

original_df['out_largest_order_size'].replace(to_replace = condition_lo,
                                value      = 1,
                                inplace    = True)

# Median Meal Rating
original_df['out_median_meal_rating'] = 0
condition_lo = original_df.loc[0:,'out_median_meal_rating'][original_df['MEDIAN_MEAL_RATING'] < out_median_meal_rating_lo]
condition_hi = original_df.loc[0:,'out_median_meal_rating'][original_df['MEDIAN_MEAL_RATING'] > out_median_meal_rating_hi]

original_df['out_median_meal_rating'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

original_df['out_median_meal_rating'].replace(to_replace = condition_lo,
                                value      = 1,
                                inplace    = True)

# Average Clicks per Visit
original_df['out_avg_clicks_per_visit'] = 0
condition_lo = original_df.loc[0:,'out_avg_clicks_per_visit'][original_df['AVG_CLICKS_PER_VISIT'] < out_avg_clicks_per_visit_lo]
condition_hi = original_df.loc[0:,'out_avg_clicks_per_visit'][original_df['AVG_CLICKS_PER_VISIT'] > out_avg_clicks_per_visit_hi]

original_df['out_avg_clicks_per_visit'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

original_df['out_avg_clicks_per_visit'].replace(to_replace = condition_lo,
                                value      = 1,
                                inplace    = True)

# Average Clicks per Visit
original_df['out_total_photos_viewed'] = 0
condition_lo = original_df.loc[0:,'out_total_photos_viewed'][original_df['TOTAL_PHOTOS_VIEWED'] < out_total_photos_viewed_lo]
condition_hi = original_df.loc[0:,'out_total_photos_viewed'][original_df['TOTAL_PHOTOS_VIEWED'] > out_total_photos_viewed_hi]

original_df['out_total_photos_viewed'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

original_df['out_total_photos_viewed'].replace(to_replace = condition_lo,
                                value      = 1,
                                inplace    = True)

# Drink Ordered
original_df['out_drink_ordered'] = 0
condition_hi = original_df.loc[0:,'out_drink_ordered'][original_df['drink_ordered'] > out_drink_ordered_hi]

original_df['out_drink_ordered'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Price
original_df['out_price'] = 0
condition_hi = original_df.loc[0:,'out_price'][original_df['out_price'] > out_price_hi]

original_df['out_price'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)
                                
                                
##############################################################################
## Feature Engineering (Trend Thresholds)                                   ##
##############################################################################

total_meals_ordered_hi = 200 # data scatters above this point
unique_meals_purch_hi = 9 # trend changes above this point
contacts_w_customer_service_hi = 10 # trend changes above this point
avg_time_per_site_visit_hi = 300 # data scatters above this point
cancellations_before_noon_hi = 8 # data scatters above this point
cancellations_after_noon_hi = 2 # data scatters above this point
weekly_plan_hi = 15 # data scatters above this point
early_deliveries_hi = 5 # data scatters above this point
late_deliveries_hi = 10 # trend changes above this point
followed_recommendations_pct_hi = 80 # trend changes above this point
avg_prep_vid_time_hi = 300 # data scatters above this point
largest_order_size_hi = 8 # trend changes above this point
master_classes_attended_hi = 2 # data scatters above this point
median_meal_rating_hi = 4.5 # trend changes above this point
avg_clicks_per_visit_hi = 10 # trend changes above this point
total_photos_viewed_hi = 700 # trend changes above this point
drink_ordered_hi = 40  # data scatters above this point
price_hi = 40 # data scatters above this point



unique_meals_purch_at = 1 # one inflated
product_categories_viewed_at = 5 # different trend at 5
cancellations_before_noon_at = 6 #trend changes at 6
pc_logins_at_lo = 3  # trend changes above this point
pc_logins_at_hi = 0 # trend changes below this point
total_photos_viewed_at = 0 # zero inflated



##### Developing Features (Columns) for Trend Changes #####

########################################
## change above threshold             ##
########################################


# Total Meals Ordered
original_df['change_total_meals_ordered'] = 0
condition = original_df.loc[0:,'change_total_meals_ordered'][original_df['TOTAL_MEALS_ORDERED'] > total_meals_ordered_hi]

original_df['change_total_meals_ordered'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Unique Meals Purchased
original_df['change_unique_meals_purch'] = 0
condition = original_df.loc[0:,'change_total_meals_ordered'][original_df['UNIQUE_MEALS_PURCH'] > unique_meals_purch_hi]

original_df['change_total_meals_ordered'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Contacts with Customer Service

original_df['change_contacts_w_customer_service'] = 0
condition = original_df.loc[0:,'change_contacts_w_customer_service'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > contacts_w_customer_service_hi]

original_df['change_contacts_w_customer_service'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)


# Cancellation Before Noon

original_df['change_cancellations_before_noon'] = 0
condition = original_df.loc[0:,'change_cancellations_before_noon'][original_df['CANCELLATIONS_BEFORE_NOON'] > cancellations_before_noon_hi]

original_df['change_cancellations_before_noon'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Cancellation After Noon

original_df['change_cancellations_after_noon'] = 0
condition = original_df.loc[0:,'change_cancellations_after_noon'][original_df['CANCELLATIONS_AFTER_NOON'] > cancellations_after_noon_hi]

original_df['change_cancellations_after_noon'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Weekly Plan

original_df['change_weekly_plan'] = 0
condition = original_df.loc[0:,'change_weekly_plan'][original_df['WEEKLY_PLAN'] > weekly_plan_hi]

original_df['change_weekly_plan'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Early Deliveries

original_df['change_early_deliveries'] = 0
condition = original_df.loc[0:,'change_early_deliveries'][original_df['WEEKLY_PLAN'] > early_deliveries_hi]

original_df['change_early_deliveries'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Late Deliveries

original_df['change_weekly_plan'] = 0
condition = original_df.loc[0:,'change_weekly_plan'][original_df['WEEKLY_PLAN'] > late_deliveries_hi]

original_df['change_weekly_plan'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Followed Recommendations Percentage

original_df['change_followed_recommendations_pct'] = 0
condition = original_df.loc[0:,'change_followed_recommendations_pct'][original_df['FOLLOWED_RECOMMENDATIONS_PCT'] > followed_recommendations_pct_hi]

original_df['change_followed_recommendations_pct'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Average Prep Video Time

original_df['change_avg_prep_vid_time'] = 0
condition = original_df.loc[0:,'change_avg_prep_vid_time'][original_df['AVG_PREP_VID_TIME'] > avg_prep_vid_time_hi]

original_df['change_avg_prep_vid_time'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Largest Order Size

original_df['change_largest_order_size'] = 0
condition = original_df.loc[0:,'change_largest_order_size'][original_df['LARGEST_ORDER_SIZE'] > largest_order_size_hi]

original_df['change_largest_order_size'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Master Class Attended

original_df['change_master_classes_attended'] = 0
condition = original_df.loc[0:,'change_master_classes_attended'][original_df['MASTER_CLASSES_ATTENDED'] > master_classes_attended_hi]

original_df['change_master_classes_attended'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Median Meal Rating

original_df['change_median_meal_rating'] = 0
condition = original_df.loc[0:,'change_median_meal_rating'][original_df['MEDIAN_MEAL_RATING'] > median_meal_rating_hi]

original_df['change_median_meal_rating'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Average Click per Visit

original_df['change_avg_clicks_per_visit'] = 0
condition = original_df.loc[0:,'change_avg_clicks_per_visit'][original_df['AVG_CLICKS_PER_VISIT'] > avg_clicks_per_visit_hi]

original_df['change_avg_clicks_per_visit'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Total Photos Viewed

original_df['change_total_photos_viewed_hi'] = 0
condition = original_df.loc[0:,'change_total_photos_viewed_hi'][original_df['TOTAL_PHOTOS_VIEWED'] > total_photos_viewed_hi]

original_df['change_total_photos_viewed_hi'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Drink Ordered

original_df['change_drink_ordered'] = 0
condition = original_df.loc[0:,'change_drink_ordered'][original_df['drink_ordered'] > drink_ordered_hi]

original_df['change_drink_ordered'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Drink Ordered

original_df['change_price'] = 0
condition = original_df.loc[0:,'change_price'][original_df['drink_ordered'] > price_hi]

original_df['change_price'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

########################################
## Change at Threshold                ##
########################################

# Total Photos Viewed

original_df['change_unique_meals_purch'] = 0
condition = original_df.loc[0:,'change_unique_meals_purch'][original_df['UNIQUE_MEALS_PURCH'] == unique_meals_purch_at]

original_df['change_unique_meals_purch'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Product Categories Viewed 

original_df['change_product_categories_viewed'] = 0
condition = original_df.loc[0:,'change_product_categories_viewed'][original_df['PRODUCT_CATEGORIES_VIEWED'] == product_categories_viewed_at]

original_df['change_product_categories_viewed'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Cancellations Before Noon

original_df['change_cancellations_before_noon'] = 0
condition = original_df.loc[0:,'change_cancellations_before_noon'][original_df['CANCELLATIONS_BEFORE_NOON'] == cancellations_before_noon_at]

original_df['change_cancellations_before_noon'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# PC Logins Low

original_df['change_pc_logins_lo'] = 0
condition = original_df.loc[0:,'change_pc_logins_lo'][original_df['PC_LOGINS'] == pc_logins_at_lo]

original_df['change_pc_logins_lo'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# PC Logins High  ##############

original_df['change_pc_logins_hi'] = 0
condition = original_df.loc[0:,'change_pc_logins_hi'][original_df['PC_LOGINS'] == pc_logins_at_hi]

original_df['change_pc_logins_hi'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)


# Total Photos Viewed

original_df['change_total_photos_viewed_at'] = 0
condition = original_df.loc[0:,'change_total_photos_viewed_at'][original_df['TOTAL_PHOTOS_VIEWED'] == total_photos_viewed_at]

original_df['change_total_photos_viewed_at'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

################################################################################
# Train/Test Split
################################################################################

# use this space to set up testing and validation sets using train/test split

# Note: Be sure to set test_size = 0.25


# Features that is used for the final model

original_df_data = ['CROSS_SELL_SUCCESS' ,
'TOTAL_MEALS_ORDERED' ,
'UNIQUE_MEALS_PURCH' ,
'CONTACTS_W_CUSTOMER_SERVICE' ,
'PRODUCT_CATEGORIES_VIEWED' ,
'AVG_TIME_PER_SITE_VISIT' ,
'MOBILE_NUMBER' ,
'CANCELLATIONS_BEFORE_NOON' ,
'CANCELLATIONS_AFTER_NOON' ,
'TASTES_AND_PREFERENCES' ,
'MOBILE_LOGINS' ,
'PC_LOGINS' ,
'WEEKLY_PLAN' ,
'EARLY_DELIVERIES' ,
'LATE_DELIVERIES' ,
'PACKAGE_LOCKER' ,
'REFRIGERATED_LOCKER' ,
'FOLLOWED_RECOMMENDATIONS_PCT' ,
'AVG_PREP_VID_TIME' ,
'LARGEST_ORDER_SIZE' ,
'MASTER_CLASSES_ATTENDED' ,
'MEDIAN_MEAL_RATING' ,
'AVG_CLICKS_PER_VISIT' ,
'TOTAL_PHOTOS_VIEWED' ,
'price' ,
'expected_drink' ,
'drink_ordered' ,
'domain_group_numeric' ,
'out_revenue' ,
'out_total_meals_ordered' ,
'out_unique_meals_purch' ,
'out_avg_time_per_site_visit' ,
'out_cancellations_before_noon' ,
'out_cancellations_after_noon' ,
'out_mobile_logins' ,
'out_pc_logins' ,
'out_weekly_plan' ,
'out_early_deliveries' ,
'out_late_deliveries' ,
'out_refrigerated_locker' ,
'out_followed_recommendations_pct' ,
'out_avg_prep_vid_time' ,
'out_largest_order_size' ,
'out_median_meal_rating' ,
'out_avg_clicks_per_visit' ,
'out_total_photos_viewed' ,
'out_drink_ordered' ,
'out_price' ,
'change_total_meals_ordered' ,
'change_unique_meals_purch' ,
'change_contacts_w_customer_service' ,
'change_cancellations_before_noon' ,
'change_cancellations_after_noon' ,
'change_weekly_plan' ,
'change_early_deliveries' ,
'change_followed_recommendations_pct' ,
'change_avg_prep_vid_time' ,
'change_largest_order_size' ,
'change_master_classes_attended' ,
'change_median_meal_rating' ,
'change_avg_clicks_per_visit' ,
'change_total_photos_viewed_hi' ,
'change_drink_ordered' ,
'change_price' ,
'change_product_categories_viewed' ,
'change_pc_logins_lo' ,
'change_pc_logins_hi' ,
'change_total_photos_viewed_at' ]

# Independent variable that needs to be predicted

original_df_target = ['REVENUE']

# Running Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(original_df[original_df_data], 
                                                    original_df[original_df_target], 
                                                    test_size = 0.25, 
                                                    random_state = 222)



################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# use this space to instantiate, fit, and predict on your final model


gbr = GradientBoostingRegressor(n_estimators=500,min_samples_split = 4,max_depth = 3, random_state = 222)

# FITTING the training data

gbr.fit(X_train,y_train.values.ravel())

# PREDICTING on new data
gbr.predict(X_test)

print('GradientBoost Model Training Score:', gbr.score(X_train, y_train).round(4))
print('GradientBoost Model Testing Score:',  gbr.score(X_test, y_test).round(4))

# saving scoring data for future use
gbr_train_score = gbr.score(X_train, y_train).round(4)
gbr_test_score = gbr.score(X_test, y_test).round(4)



################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test)

test_score = gbr.score(X_test, y_test).round(4)




"""

elapsed_time = timeit.timeit(code_to_test, number=3)/3
print(f"""{'*' * 40}Code Execution Time (seconds): {elapsed_time}{'*' * 40}""")


# In[ ]:




