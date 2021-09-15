def calc_trans_by11month(df_train):
    '''
    ?Func считает суммы транзакций за каждый месяц по всем юзерам

    IN: df_train :: очищенный
    OUT: cat_ptable [{юзер,месяц} сумма_месяца_Категория1,...,сумма_месяца_Категория37] 

    !Код автоматом удаляет наблюдения за последний месяц (1ый масимальный) из сета
    !OUT cat_ptable: содержит NaN значения в columns = сумма_месяца_Категория1,...,сумма_месяца_Категория37
    !OUT cat_ptable: party_rk повторяются в rows
    !OUT cat_ptable: len(cat_ptable) != 50 000, так как не все пользователи совершали транзаккции в clean df_train
    '''

    #1 ТУТА я делаю создаю вспомогательный столбец = первый день месяца в котором произошла операция
    df = df_train[['transaction_dttm','party_rk','transaction_amt_rur','category']]
    df['transaction_dttm'] = pd.to_datetime(df['transaction_dttm'] )
    df['month_dttm'] = df['transaction_dttm'] + pd.offsets.Day() - pd.offsets.MonthBegin()

    #2 ТУТА я убираю все наблюдения последнего месяца
    df = df[df['month_dttm'] < max(df['month_dttm'])]

    #3 ТУТА я делаю группировку сум_транзакций по месяцам
    df = df.groupby(['party_rk','category','month_dttm']).sum()
    df = df.reset_index()
    
    #4 ТУТА я делаю cat_ptable 
    cat_ptable = pd.pivot_table(df, values=['transaction_amt_rur'], columns=['category'],  index=['party_rk','month_dttm'], aggfunc={np.sum})
    cat_ptable.columns = cat_ptable.columns.get_level_values(2)
    cat_ptable = cat_ptable.reset_index()
    return cat_ptable

def calc_kamron_features(cat_ptable,df_socdem):
    '''
    ?Считает features 

    IN: cat_ptable: from calc_trans_by11month(), df_socdem
    OUT: df_kamron_features : for every user of 50 000 users

    !Features считаются для 11 месяцев
    '''
    #1 ТУТА я заполняю дефолтными np.NaN суммы_расходов_категорий пользователей, которые пропали после чистки или удаления 12го месяца 
    #   в итоге должно получиться len(df_socdem) == 50 000 штук наблюдений 
    lost_users = list(set(df_socdem.party_rk)-set(cat_ptable.party_rk))
    row = cat_ptable.iloc[:1] 
    for user_id in lost_users:
        row.party_rk = user_id
        row.iloc[:,2:row.shape[1]] = np.NaN
        cat_ptable = cat_ptable.append(row, ignore_index = True)
    
    grouped_ptabel = cat_ptable.groupby(['party_rk'])
    #2 ТУТА считаю кол-во месяцев с операциями OUT:feature_count
    feature_count = grouped_ptabel.count()
    # ставлю суфикс '_count'в columns
    feature_count.columns = pd.Index(list([str(i)+'_count' for i in list(feature_count.columns)]))
    
    grouped_ptabel = cat_ptable.fillna(0).groupby(['party_rk'])
    #3 ТУТА считаю среднее операций OUT:feature_mean
    feature_mean = grouped_ptabel.mean()
    # ставлю суфикс '_mean'в columns
    feature_mean.columns = pd.Index(list([str(i)+'_mean' for i in list(feature_mean.columns)]))

    #4 ТУТА считаю среднее операций OUT:feature_std
    feature_std = grouped_ptabel.mean()
    # ставлю суфикс '_std'в columns
    feature_std.columns = pd.Index(list([str(i)+'_std' for i in list(feature_std.columns)]))

    #5 Объеденяю в одну таблицу 
    feature_all = (feature_count
                                .reset_index().merge(feature_mean.reset_index(), on='party_rk', how='left')
                                              .merge(feature_std.reset_index(), on='party_rk', how='left'))
    return feature_all

# ПРИМЕР КОДА, КОТОРЫЙ СЧИТАЕТ FEATURES
# cat_ptable = calc_trans_by11month(df_train)
# df_kamron_final = cat_features = calc_kamron_features(cat_ptable,df_socdem)
# df_kamron_final