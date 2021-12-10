# ### LDR; put this after the final function is defined 

# #%%
# x_test

# #%%
# ###############################################################################
# ## testing the final function
# ###############################################################################
# # test_list = independent_vars[]
# testing_df = DATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf']]
# test_wop, test_cpt = predict_LDR(testing_df.loc[437275].to_list())
# print("wop: " + str(test_wop) + " \ncpt: " + str(test_cpt))

# #%%
# ## what the predicted WOP is supposed to be: 0.6174995841247563
# x_train
# predicted_wop.item(0)

# #%%
# ## step through the function line by line
# ## make list input into a dataframe
# list = DATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf']].loc[437275].to_list()
# input_df = pd.DataFrame(columns = ['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf'])
# input_df.loc[0] = list
# ## one hot encoding
# oneHotEncoding = pd.get_dummies(input_df['i_sex'])
# input_df = input_df.drop('i_sex', axis = 1)
# input_df = input_df.join(oneHotEncoding)
# input_df = input_df.rename(columns={1:"i_sex_1", 2:"i_sex_2", 3:"i_sex_3"})
# # # normalize data using cleaned table from above 
# cleaned_dropped = cleaned.drop(columns=["CPT", "WOP"])
# input_df = (input_df - cleaned_dropped.mean()) / cleaned_dropped.std()
# input_df = input_df.fillna(x_test["i_sex_1"].to_list()[0])
# input_df = input_df.reindex(sorted(input_df.columns), axis=1) ## input_df stuff all looks fine
# input_df
# # x_test.loc[0] = input_df.to_numpy().tolist()[0]
# # x_test.sort_index
# # np.sum((input_df.to_numpy() * wop_coef)[0][:7])


# predict_wop = final_wop_alg.predict(input_df)
# predict_wop[0]
# # predict_wop.item(0) ## why is this 128027.71426319 even though the values are the same as the x_test values?
# # if predict_wop > 1:
# #     predict_wop = 1
# # predict_wop =  np.exp(predict_wop)
# # predict_cpt = final_cpt_alg.predict(input_df).item(0)
# # predict_cpt
# # if predict_cpt < -1:
# #     predict_cpt = -1
# # ## turn back into origianl units 
# # wop, cpt =  predict_wop * cleaned["WOP"].std() + cleaned["WOP"].mean(), predict_cpt * cleaned["CPT"].std() + cleaned["CPT"].mean()
# # print(str(predict_wop) + str(predicted_cpt))
# # print(str(wop) + str(cpt))

# # %%




# #%%
# x_test

# #%%
# ###############################################################################
# ## testing the final function
# ###############################################################################
# # test_list = independent_vars[]
# testing_df = DATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf']]
# test_wop, test_cpt = predict_SDR(testing_df.loc[8112].to_list())
# print("wop: " + str(test_wop) + " \ncpt: " + str(test_cpt))

# #%%
# ## what the predicted WOP is supposed to be: -0.15884896138347057
# x_train
# predicted_wop.item(0)

# #%%
# ## step through the function line by line
# ## make list input into a dataframe
# list = DATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf']].loc[8112].to_list()
# input_df = pd.DataFrame(columns = ['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf'])
# input_df.loc[0] = list
# ## one hot encoding
# oneHotEncoding = pd.get_dummies(input_df['i_sex'])
# input_df = input_df.drop('i_sex', axis = 1)
# input_df = input_df.join(oneHotEncoding)
# input_df = input_df.rename(columns={1:"i_sex_1", 2:"i_sex_2", 3:"i_sex_3"})
# # # normalize data using cleaned table from above 
# cleaned_dropped = cleaned.drop(columns=["CPT", "WOP"])
# input_df = (input_df - cleaned_dropped.mean()) / cleaned_dropped.std()
# input_df = input_df.fillna(x_test["i_sex_1"].to_list()[0])
# input_df = input_df.reindex(sorted(input_df.columns), axis=1) ## input_df stuff all looks fine
# input_df
# # x_test.loc[0] = input_df.to_numpy().tolist()[0]
# # x_test.sort_index
# # np.sum((input_df.to_numpy() * wop_coef)[0][:7])


# # predict_wop = final_wop_alg.predict(input_df)
# # predict_wop.item(0) ## why is this 128027.71426319 even though the values are the same as the x_test values?
# # if predict_wop > 1:
# #     predict_wop = 1
# # predict_wop =  np.exp(predict_wop)
# # predict_cpt = final_cpt_alg.predict(input_df).item(0)
# # predict_cpt
# # if predict_cpt < -1:
# #     predict_cpt = -1
# # ## turn back into origianl units 
# # wop, cpt =  predict_wop * cleaned["WOP"].std() + cleaned["WOP"].mean(), predict_cpt * cleaned["CPT"].std() + cleaned["CPT"].mean()
# # print(str(predict_wop) + str(predicted_cpt))
# # print(str(wop) + str(cpt))






# #### SDR final model testing 

# #%%
# x_test

# #%%
# ###############################################################################
# ## testing the final function
# ###############################################################################
# # test_list = independent_vars[]
# testing_df = DATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf']]
# test_wop, test_cpt = predict_SDR(testing_df.loc[8112].to_list())
# print("wop: " + str(test_wop) + " \ncpt: " + str(test_cpt))

# #%%
# ## what the predicted WOP is supposed to be: -0.15884896138347057
# x_train
# predicted_wop.item(0)

# #%%
# ## step through the function line by line
# ## make list input into a dataframe
# list = DATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf']].loc[8112].to_list()
# input_df = pd.DataFrame(columns = ['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf'])
# input_df.loc[0] = list
# ## one hot encoding
# oneHotEncoding = pd.get_dummies(input_df['i_sex'])
# input_df = input_df.drop('i_sex', axis = 1)
# input_df = input_df.join(oneHotEncoding)
# input_df = input_df.rename(columns={1:"i_sex_1", 2:"i_sex_2", 3:"i_sex_3"})
# # # normalize data using cleaned table from above 
# cleaned_dropped = cleaned.drop(columns=["CPT", "WOP"])
# input_df = (input_df - cleaned_dropped.mean()) / cleaned_dropped.std()
# input_df = input_df.fillna(x_test["i_sex_1"].to_list()[0])
# input_df = input_df.reindex(sorted(input_df.columns), axis=1) ## input_df stuff all looks fine
# input_df
# # x_test.loc[0] = input_df.to_numpy().tolist()[0]
# # x_test.sort_index
# # np.sum((input_df.to_numpy() * wop_coef)[0][:7])


# # predict_wop = final_wop_alg.predict(input_df)
# # predict_wop.item(0) ## why is this 128027.71426319 even though the values are the same as the x_test values?
# # if predict_wop > 1:
# #     predict_wop = 1
# # predict_wop =  np.exp(predict_wop)
# # predict_cpt = final_cpt_alg.predict(input_df).item(0)
# # predict_cpt
# # if predict_cpt < -1:
# #     predict_cpt = -1
# # ## turn back into origianl units 
# # wop, cpt =  predict_wop * cleaned["WOP"].std() + cleaned["WOP"].mean(), predict_cpt * cleaned["CPT"].std() + cleaned["CPT"].mean()
# # print(str(predict_wop) + str(predicted_cpt))
# # print(str(wop) + str(cpt))

# # %%
