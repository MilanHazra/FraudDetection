# -*- coding: utf-8 -*-
'''*************************************************
: import package
*************************************************'''
import pymysql
import pandas as pd
import os

'''*************************************************
: Create db connect
*************************************************'''
def get_connection():
    connection=pymysql.connect(host='127.0.0.1',
                               port=3306,
                               user='root',
                               password='root',
                               db='local_excel_files',
                               charset='utf8',
                               cursorclass=pymysql.cursors.DictCursor)
    return connection

'''#call get_connection()'''
get_connection()
'''close connection'''
def close_connection(conn):
    conn.close
    
'''get data from db '''
def get_dbdata(sql:str):
    conn=get_connection()
    data=pd.read_sql(sql,conn)
    close_connection(conn)
    return data
'''*************************************************
        query
*************************************************'''


'''1: show all table in db'''
def show_table():
    sql='''show tables'''
    return get_dbdata(sql)
show_table()

'''2: query for  required '''
def sample_fund5_example():
    sql='''select * from fund5_example_data_for_dt'''
    #sql=SELECT * FROM owner_2018q2_broaderrussel WHERE fund_id=5
    return get_dbdata(sql)
#data=sample_fund5_example()