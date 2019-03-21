columns = ["TESTId", "Code", "Native_Heap", "System", "Private_Others", "Graphics", "Java_Heap", "Stack"]
test_rows = {"Airtel_PinScreen": 2, "Airtel_HomeScreen": 3, "Airtel_DetailScreen": 4,
     "Airtel_PlayerScreen": 5,
     "Airtel_Scrolling": 6, "Airtel_SearchScreen": 7}
train_rows = {"Netflix_PinScreen": 2, "Netflix_HomeScreen": 3, "Netflix_DetailScreen": 4,
     "Netflix_PlayerScreen": 5,
     "Netflix_Scrolling": 6, "Netflix_SearchScreen": 7}


# df_train_x = train_x.describe()
# df_test_x = test_x.describe()
# print df_train_x
#
# std_dict = {}
# for i in range(0, 6):
#     std_dict[columns[i+1]] = [df_train_x.iloc[2, i], df_test_x.iloc[2, i]]
# print std_dict
#
# for i in range(1, 7):
#     plt.subplot(2, 3, i)
#     plt.plot(std_dict[columns[i]][0], std_dict[columns[i]][1], 'ro')
#     plt.title(columns[i])