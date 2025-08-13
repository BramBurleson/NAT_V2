import pandas as pd
#make lists here:
# sub-01  : Krys          : 20250411              : yes : good
# sub-02  : Georgina      : 20250513+20250520     : yes : good
# sub-03  : Crystal       : 20250520              : yes : bad left screen, flicker runs 05,06,07 +  localizer
# sub-04  : Maya          : 20250522              : O : ?
# sub-05  : Elijah        : 20250522              : ? : ?
# sub-06  : Annalise      : 20250527              : ? : ?
# sub-07  : Teresa        : 20250530              : ? : ?
# sub-08  : Anastasiia    : 20250530              : 0 : ?
#add paths to subject notes.
subject_info = pd.DataFrame({
'subject_ids' : ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08'],
'subject_names' : ['Krys', 'Georgina', 'Crystal', 'Maya', 'Elijah', 'Annalise', 'Teresa', 'Anastasiia'],
'scan_dates': ['20250411', '20250513_20250520', '20250520', '20250522', '20250522', '20250527', '20250530', '20250530'],
'status' : ['completed','completed','completed','completed','today','','',''],
'TaskFF': ['yes','yes','yes','yes','yes','yes','yes','yes'],
'TaskFF_ResponseWindow [ms]': ['2000','2000','2000','1500','1500','','',''],
'T1': ['yes','yes','yes','yes','yes','','',''],
'Task_NAT': ['yes','yes','yes','yes','yes','yes','yes','yes'],
'scan_quality_recap': ['good','good','bad left screen, flicker runs 05,06,07 +  localizer','?','?','?','?','?'],
})

subject_info.to_csv(r"C:\Users\bramb\Desktop\20250527_subject_info.csv")