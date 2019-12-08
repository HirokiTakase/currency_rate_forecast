import datetime
import pandas as pd

# 日付から曜日を判定
def get_weekday(date):
    weekday = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    # d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    return weekday[date.weekday()]

# weekend_exclude = trueで週末を加えない
def make_weekday_array(start, end, weekend_exclude):
   
    start_temp_for = start + datetime.timedelta(days = 1)
    start_temp_for = datetime.datetime.strftime(start_temp_for,'%Y-%m-%d 00:00:00')
    start_temp_for = datetime.datetime.strptime(start_temp_for,'%Y-%m-%d %H:%M:%S')
    start_temp = datetime.datetime.strftime(start,'%Y-%m-%d 23:59:00')
    start_temp = datetime.datetime.strptime(start_temp,'%Y-%m-%d %H:%M:%S')
    end_temp_for = datetime.datetime.strftime(end,'%Y-%m-%d 00:00:00')
    end_temp_for = datetime.datetime.strptime(end_temp_for, '%Y-%m-%d %H:%M:%S')

    weekday_array_start = []
    weekday_array_end = []

    if weekend_exclude == True:
        if get_weekday(start) != 'Sat' or get_weekday(start) != 'Sun':
            weekday_array_start.append(start) 
            weekday_array_end.append(start_temp)

        while(1) : 
            if get_weekday(start_temp_for) == 'Fri':
                weekday_array_start.append(start_temp_for)
                weekday_array_end.append(start_temp_for + datetime.timedelta(days = 1) - datetime.timedelta(minutes= 1))
            elif get_weekday(start_temp_for) == 'Sat' or get_weekday(start_temp_for) == 'Sun':
                pass 
            else :
                weekday_array_start.append(start_temp_for)
                weekday_array_end.append(start_temp_for + datetime.timedelta(days = 1) - datetime.timedelta(minutes= 1))
            
            if start_temp_for == end_temp_for :
                break

            start_temp_for = start_temp_for + datetime.timedelta(days = 1)

        if (get_weekday(end) == 'Sat') or (get_weekday(end) == 'Sun'):
            pass
        else:
            weekday_array_start.append(end_temp_for) 
            weekday_array_end.append(end)
    else :
        
        
        weekday_array_start.append(start) 
        weekday_array_end.append(start_temp)

        while(1) : 
            
            weekday_array_start.append(start_temp_for)
            weekday_array_end.append(start_temp_for + datetime.timedelta(days = 1) - datetime.timedelta(minutes= 1))
    
            if start_temp_for == end_temp_for :
                break

            start_temp_for = start_temp_for + datetime.timedelta(days = 1)
        
        if end != end_temp_for:
            weekday_array_start.append(end_temp_for) 
            weekday_array_end.append(end)

    return  pd.DataFrame({'Start' : weekday_array_start,'End' : weekday_array_end})
    
    
if __name__ == "__main__":
    make_weekday_array(datetime.datetime(2019,10,31,3,45),datetime.datetime(2019,11,15,8,1,0),null)