#datacamp #data_analysis   
# Dates in Calendars
## Dates in Python
python has a separate class for dates.
```python
from datetime import date
dates = date(2019, 7, 8) -> date(year, month, day)
print(dates.year) -> gives the year 2019
print(dates.month) -> gives the month 7(July)
print(dates.day) -> gives the day 8
print(dates.weekday()) -> gives the day of the week 0(Monday)
```
the weekdays are given as
Monday -> 0
Tuesday -> 1
Wednesday -> 2
Thursday -> 3
Friday -> 4
Saturday -> 5
Sunday -> 6

## Math with Dates
just like numbers we can even do some similar operations with dates
```python
from datetime import date
from datetime import timedelta
d1 = date(2019, 7, 13)
d2 = date(2019, 8, 21)
l = [d2, d1]
min(l) -> will return minimum date i.e. early date d1
max(l) -> will return maximum date i.e. later date d2
td = d1-d2 -> returns a timedelta object thats stores info like how many days passed
td.days -> returns the days passed
td = timedelta(days=209)
td+d1 returns a new date after 209 days
```
you can't add dates btw

## Dates into Strings
```python
d = date(2019, 6, 7)
print(d) -> #will print 2019-06-07 as a string
```
this is the default ISO 8601 format. the format is "YYYY-MM-DD"
to get the ISO format we can also use 
`d.isoformat()`
ISO format is useful for sorting dates. no other format can help because it comes in descending order that is year then month then day
to change the format into anything we want we can use
```python
print(d.strftime("%Y-%d-%m"))
```
you can write whatever you want in the `strftime` method because its very flexible.
Just remember that 
1. %Y is year 
2. %m is month
3. %d is day
4. %B is the month in words
5. %j is the day of the year

# Dates and Time
## Using time with dates
```python
from datetime import datetime
dt = datetime(2019, 10, 1, 15, 23, 25) -> october 1st 2019, at 3:23:25PM
datetime(year, month, day, hour, minute, second, microsecond)
dt.replace(minute=0, second=0, microsecond=0) -> replaces values
```

if you want to add .5 seconds, add microseconds=5000
printing as `dt.isoformat()` will give the date and then a T then the time.
just like date we can access the minutes, hours with `.minute` or `.hour` 

## Parsing dates
just like with dates `strftime` can be used to print the `datetime` object in any format
to parse dates of any format we use `.strptime`
```python
dt = datetime.strptime("2019/12/10 13:13:13", "%Y/%m/%d %H:%M:%S")
```
the first one is the input and the second one is the format of the input.
1. %H is Hour
2. %M is Minute
3. %S is second
###### Fun Fact
there is something known as the UNIX timestamp. this is when computers record dates as the number of seconds that has passed since January 1st , 1970
to parse these
```python
ts = 1514665153.0
print(datetime.fromtimestamp(ts))
```
prints 30/12/2017 15:19:13

## Durations
when we subtract two `datetime` objects we get a `timedelta` object, which contains the duration/ the time elapsed
```python
duration = d1 - d2
print(duration.total_seconds()) ->prints the total seconds

from datetime import timedelta
td = timedelta(seconds=1, days=1, weeks=-1)
d1 + td -> gives a new datetime with the previous week and a day and a second later
```
`timdelta` objects are basically durations and can be added or subtracted to `datetime` objects

# Time zones and Daylight Savings
## UTC offsets
people usually set their clocks to a time such that, when the sun was directly overhead it would be 12PM. however, this meant clocks varied a lot around lots of places even though if by just 20 - 30 minutes. so it was decided that every large area will have a specific time, even if some places in that area actually have the slightly wrong time. this was first done by the United Kingdom. so, all other time zones are relative to the UK standard known as UTC.
![[Pasted image 20231128143809.png]]
west usually means they are early to UTC and east means later.
```python
from datetime import datetime, timezone, timedelta
ET = timezone(timedelta(hours=-5)) -> USA time according to UTC is UTC-5
#this creates timezone UTC-5

dt = datetime(2019, 1, 1, 15, 9, 3, tzinfo=ET)
#this sets the timezone of the recorded clock as UTC-5
```
now when you print the datetime object, it prints the date, time and the offset as `-05:00`
to see the time in a different time zone, we use
```python
IST = timezone(timedelta(hour=5, minutes=30))
print(dt.astimezone(IST))
```
this will print the date in terms of Indian Standard Time and the offset of `+05:30`
being "time zone naive" means the data doesn't have a time zone.
when you use the `replace` method to change time zone, only the offset changes. that means you just changed the time zone you measured that time in.
however, if you use `astimezone` it will convert the given time to the required time zone 
```python
print(dt.replace(tzinfo=timezone.utc)) #prints "2019-01-01 15:09:03+00:00"
print(dt.astimezone(timezone.utc)) #prints "2019-01-01 20:09:03+00:00" 
```

## Time zone Database
```python
from datetime import datetime
from dateutil import tz
et = tz.gettz('Continent/City')
dt = datetime(2019, 3, 3, 15, 3, 9, tzinfo=et)
```
the `tz` package is the up-to-date package containing the time zones across all regions. to get the time zone we use  `tz.gettz("America/New_York")`
this database also takes care of daylight savings

## Forward Daylight savings
some places change their clock twice a year, to create longer summer evenings -> this is known as daylight saving. first, the clocks are moved forward during the spring.
this is done by changing the UTC offset at that exact moment
At 12-03-2017 at 1:59:59 the offset is still -5 hours in Washington, DC
but after that the time becomes 3:00:00 and the offset is -4 hours.
to show this, 
```python
EST = timezone(timedelta(hours=-5))
EDT = timezone(timedelta(hours=-4))
dt1 = datetime(2017, 12, 3, 1, 59, 59, tzinfo=EST)
dt2 = datetime(2017, 12, 3, 3, 0, 0, tzinfo=EDT)
dt2-dt1 -> this will result in 1 second

from dateutils import tz
et = tz.gettz("America/New_York")
dt1 = datetime(2017, 12, 3, 1, 59, 59, tzinfo=et)
dt2 = datetime(2017, 12, 3, 3, 0, 0, tzinfo=et)
dt2-dt1 -> 1 second
```
basically what happens is that, 
at dt1, the time in UTC is 6:59:59 AM. 
at dt2, the time in UTC is 7:00:00 AM.
the difference is 1 second. that is how daylight savings are calculated. we just change the offset.

## Resetting Daylight savings
on 5/11/2017 the times were reset. that is after 1:59:59 it went back to 1:00:00.
this is done by resetting the UTC offset back to -5
```python
et = tz.gettz("America/New_York")
first_1am = datetime(2017, 11, 5, 1, 0, 0, tzinfo=et)
tz.datetime_ambiguous(first_1am)
#returns true meaning that timing is ambiguous and occurs multiple times

second_1am = datetime(2017, 11, 5, 1, 0, 0, tzinfo=et)
second_1am = tz.enfold(second_1am)
#marks that this is the second occurrence of that time

second_1am - first_1am ->results in zero tho. but,
first_1am = first_1am.astimezone(tz.utc)
second_1am = second_1am.astimezone(tz.utc)
second_1am - first_1am -> gives 60 minutes
```
moral is to always use the UTC time to calculate absolute intervals
***If you are collecting data always store in UTC format or at a constant UTC offset***

# Datetime in Pandas
## Reading date and time in pandas
```python
df = pd.read_csv("csv_file.csv", parse_dates=["date cols"])
or
df["date col"] = pd.to_datetime(df["date col"], format="%Y-%m-%d %H:%M:%S")
```
You can either convert them to datetime using `parsedates` while importing or after using `to_datetime()`  where you specify the format. then you can do most of the thing you did in the previous sections and even methods on the columns with `.dt` attribute.

## Summarizing datetime columns
we can use the methods mean, and sum on `timedelta` columns.
to group by datetime columns we use the `resample` method
```python
df.resample("M", on="date col")["any col"].aggregating function like mean
example
rides.resample("M", on="Start Date")["duration"].mean()
```
using "M" means the columns are grouped by the month attribute.
we can also plot using resample
```python
df.resample("D", on="date col")["any col"].mean().plot()
```
![[Pasted image 20231128181937.png]]

## Methods on Pandas datetime columns
to set the time zone we use
```python
df[date col] = df[date col].dt.tz_localize("America/New_York", ambiguous="NaT")
```
this method will give an error if there are ambiguous dates. due to daylight savings
so, we add an extra parameter
`NaT` means Not a Time.
pandas skips `NaT` rows when doing methods.
`df[date col].dt.day_name()` -> gives the name of the day
`df[date col].shift(1)` means we shift the date col by one meaning the first date will be a `NaT`.
that is the column is "down-shifted". and the first row is a `NaT`
this is useful when comparing start and end dates.
`df[date col].dt.tz_convert("continent/city")`
converts the column to a given time zone.