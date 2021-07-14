import pandas as pd
import functools
import time

def timer(func):
        """Print the runtime of the decorated function."""
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = start_time - end_time
            print(f"Finished {func.__name__!r} in {-run_time:.4f} s")
            return value
        return wrapper_timer

@timer
def PatValues(data):
    Pat_ID_Values = data['Patient ID'].value_counts().sort_index()
    UniquePat = len(Pat_ID_Values) 
    
    return Pat_ID_Values, UniquePat

# def PatSequences(data, UniquePat):
#     UniqPatSeq = data.value_counts(['Patient ID', 'Seq ID']).sort_index()

#     SeqPerPat = []
#     for i in range(1, UniquePat + 1):
#         SeqPerPat.append(len(UniqPatSeq[i]))
    
#     return SeqPerPat, UniqPatSeq

@timer
def PatSequences(data, UniquePat):
    UniqPatSeq = data.value_counts(['Patient ID', 'Seq ID']).sort_index()
    SeqPat = data.groupby("Patient ID")["Seq ID"].nunique()
    
    SeqPerPat = []
    for i in range(UniquePat):
        SeqPerPat.append(SeqPat.iloc[i])
    
    return SeqPerPat, UniqPatSeq

# def FrameIndexingByPat(UniquePat, SeqPerPat, UniqPatSeq):
#     counter = 0
#     IndexByPatient = []
#     for i in range(UniquePat):    
#         CurrentGroup = []
#         for j in range(SeqPerPat[i]):
#             CurrentGroup1=[]
#             for k in range(UniqPatSeq[i+1][j]):
#                 CurrentGroup1.append(counter)
#                 counter +=1
#             CurrentGroup.append(CurrentGroup1)        
#         IndexByPatient.append(CurrentGroup) 
        
#     return IndexByPatient

@timer
def FrameIndexingByPat(UniquePat, SeqPerPat, UniqPatSeq):
    counter = 0
    counter2 = 0
    IndexByPatient = []
    for i in range(UniquePat):    
        CurrentGroup = []
        for j in range(SeqPerPat[i]):
            CurrentGroup1=[]
            for k in range(UniqPatSeq.iloc[counter2]):
                CurrentGroup1.append(counter)
                counter +=1
            CurrentGroup.append(CurrentGroup1)
            counter2 += 1
        IndexByPatient.append(CurrentGroup) 
        
    return IndexByPatient

@timer
def NumSequences(UniquePat, SeqLen, IndexByPatient, SeqPerPat):
    NumOfSequences = []
    for i in range (UniquePat):
        CurrentGroup = []
        for j in range(SeqPerPat[i]):
            Num = len(IndexByPatient[i][j]) - (SeqLen - 1)
            CurrentGroup.append(Num)
        NumOfSequences.append(CurrentGroup)  
        
    return NumOfSequences

@timer
def SlidingWindow(UniquePat, SeqLen, NumOfSequences, IndexByPatient):
    IndexedWindow = []
    for i in range(UniquePat):
        for j in range(len(NumOfSequences[i])):
            for k in range(NumOfSequences[i][j]):
                CurrentGroup = []
                for l in range(SeqLen):
                    CurrentGroup.append(IndexByPatient[i][j][k+l])
                IndexedWindow.append(CurrentGroup)
                
    return IndexedWindow

@timer
def SlidingWindowV2(UniquePat, SeqLen, NumOfSequences, IndexByPatient):
    IndexedWindow = []
    for i in range(UniquePat):
        for j in range(len(NumOfSequences[i])):
            for k in range(NumOfSequences[i][j]):
                CurrentGroup = []
                # for l in range(SeqLen):
                CurrentGroup.append(IndexByPatient[i][j][k:k+SeqLen])
                IndexedWindow.append(CurrentGroup)
                
    return IndexedWindow

 # r"C:\Users\sahil\Documents\Patient Data\metadata.csv"
@timer
def main():
    
    df = pd.read_csv(r"C:\Users\sahil\Documents\Patient Data\metadata.csv")
    SeqLen = 300
    
    # Pat_ID_Values = df['Patient ID'].value_counts().sort_index()
    # UniquePat = len(Pat_ID_Values)
    
    UniqPatSeq = df.value_counts(['Patient ID', 'Seq ID']).sort_index()
    print(UniqPatSeq)
    # # print(UniqPatSeq.iloc[8])
    
    # # print(len(UniqPatSeq[1]))
    
    # # print(UniqPatSeq.iloc[0])
    
    # x = df.groupby("Patient ID")["Seq ID"].nunique()
    # # print(x)
    # # USE THIS
    # # Use iloc for indexing issues
  
    # SeqPerPat = []
    # for i in range(UniquePat):
    #     SeqPerPat.append(x.iloc[i])
        
    # print(SeqPerPat[0])


    # counter = 0
    # counter2 = 0
    # IndexByPatient = []
    # for i in range(UniquePat):    
    #     CurrentGroup = []
    #     for j in range(SeqPerPat[i]):
    #         CurrentGroup1=[]
    #         for k in range(UniqPatSeq.iloc[counter2]):
    #             CurrentGroup1.append(counter)
    #             counter +=1
    #         CurrentGroup.append(CurrentGroup1)
    #         counter2 += 1
    #     IndexByPatient.append(CurrentGroup) 
        
    # print(IndexByPatient)
    

    PatVals, NumPat = PatValues(df)

    SeqPerPat, UniqPatSeq = PatSequences(df, NumPat)
    
    IndexedPatients = FrameIndexingByPat(NumPat, SeqPerPat, UniqPatSeq)

    NumOfSequences = NumSequences(NumPat, SeqLen, IndexedPatients, SeqPerPat)
   
    IndexedWindow = SlidingWindow(NumPat, SeqLen, NumOfSequences, IndexedPatients)  

    # IndexedWindowV2 = SlidingWindowV2(NumPat, SeqLen, NumOfSequences, IndexedPatients)     


    print(IndexedWindow)
    # print(IndexedWindowV2)
    print(len(IndexedWindow))
    # print(len(IndexedWindowV2))
    
    
    
    
    # print(IndexedWindow == IndexedWindowV2)




if 1==1:
    main()




# make a for loop appending each patient number into an array - use that array as the range in PatSequences







# df = pd.read_csv("Random.csv")
# SeqLen = 6




# Pat_ID_Values = df['Patient ID'].value_counts().sort_index()
# UniquePat = len(Pat_ID_Values)

# print(UniquePat)

# UniqPatSeq = df.value_counts(['Patient ID', 'Seq ID']).sort_index()

# SeqPerPat = []
# for i in range(1, UniquePat + 1):
#     SeqPerPat.append(len(UniqPatSeq[i]))               
    
# counter = 0
# IndexByPatient = []
# for i in range(UniquePat):    
#     CurrentGroup = []
#     for j in range(SeqPerPat[i]):
#         CurrentGroup1=[]
#         for k in range(UniqPatSeq[i+1][j]):
#             CurrentGroup1.append(counter)
#             counter +=1
#         CurrentGroup.append(CurrentGroup1)        
#     IndexByPatient.append(CurrentGroup)       
    
# NumOfSequences = []
# for i in range (UniquePat):
#     CurrentGroup = []
#     for j in range(len(IndexByPatient[i])):
#         Num = len(IndexByPatient[i][j]) - (SeqLen - 1)
#         CurrentGroup.append(Num)
#     NumOfSequences.append(CurrentGroup)    

# SlidingWindow = []
# for i in range(UniquePat):
#     for j in range(len(NumOfSequences[i])):
#         for k in range(NumOfSequences[i][j]):
#             CurrentGroup = []
#             for l in range(SeqLen):
#                     CurrentGroup.append(IndexByPatient[i][j][k+l])
#             SlidingWindow.append(CurrentGroup)


# print(SlidingWindow)
# print(len(SlidingWindow))





# Gets data corresponding to columns that are named in the argument {}
# data = {"Patient ID", "Frame Number", "Seq ID"}

# Gets the raw number of frames corresponding to each Seq ID
# Length therefore gives number of unique Seq IDs
# Seq_ID_Values = df['Seq ID'].value_counts().sort_index()
# UniqueIds = len(Seq_ID_Values)


'''
# Returns number of rows - this corresponds to the number of frames
count_rows = df.shape[0]
count_cols = df.shape[1]

# Returns list of Frame Numbers
FrameNumbers = df.loc[:,"Frame Number"]
# SeqNumbers = df.loc[['Patient ID', 'Seq ID']]


# Loop that groups all the frames in a single Seq ID into a single array
# EX: If Seq ID #0 has frames 1-10, said frames will be in one array
#     If Seq ID #1 has frames 11-18, said frames will be in another array. 
# All arrays will be stored within one big array


# This loop makes an array with each entry corresponding to the number of frames in each Seq ID
NumOfFramesPerID = []
for i in range(UniqueIds):
    ID = df['Seq ID'].value_counts()[i]
    NumOfFramesPerID.append(ID)

# This loop makes an array with each entry being an array with frames corresponding to each Seq ID
FramesBySeqID = []
count = 0
for i in range(UniqueIds):
    CurrentSet = []
    for j in range(NumOfFramesPerID[i]):
        CurrentSet.append(FrameNumbers[count])
        count += 1
    FramesBySeqID.append(CurrentSet)    

SeqLen = 6

# Loop that makes an array with each entry corresponding to the number of possibles sequences in each Seq ID
NumOfSequences = []
for i in range(UniqueIds):
    Num = len(FramesBySeqID[i]) - (SeqLen - 1)
    NumOfSequences.append(Num)


# This is the main function that is responsible for grouping the frames in a 
# Sliding window method based on the set SeqLen
# Loop groups each SeqID seperatley 
SlidingWindow = []
for i in range (UniqueIds):
    for j in range(NumOfSequences[i]):
        CurrentSet=[]
        for k in range(SeqLen):
            CurrentSet.append(FramesBySeqID[i][j+k])
        SlidingWindow.append(CurrentSet)
        
# print(SlidingWindow)

# IT FUCKING WORKS LETS FUCKING GO



# NEXT STEPS
# make a system to group frames - DONEEEEEEEEEEEEEEE
# make system to account for patient ID - THIS IS THE HARD PART

       
    
    # group by seq id and patient id
        


'''







# MAIN LOOP





# d = df['Seq ID'].value_counts()[0]
# print(d)










# seq_len = 6

# count_rows = df.shape[0]
# count_cols = df.shape[1]

# NumberOfGroups = count_rows - (seq_len - 1)

# print(NumberOfGroups)

# length = df['Seq ID'].value_counts().sort_index()
# print(length)
# d = df['Seq ID'].value_counts()[0]
# print(d)

# for i in range (NumberOfGroups):
#     pass






# FrameNumbers = df.loc[:,"Frame Number"]
# SeqNumbers = df.loc[:, "Seq ID"]

# NumberOfSequences = 1
# for i in range(1, count_rows - 1):
#     Seq_ID = df.loc[i, "Seq ID"]
#     Seq_ID2 = df.loc[i-1, "Seq ID"]

#     if Seq_ID != Seq_ID2:
#         NumberOfSequences += 1
 
# counter = 0
# Seq_Frames = [[1,2,3,4,5,6], [7,8,9,10,11]]
# for i in range(1, count_rows - 1):
#     Seq_ID = df.loc[i, "Seq ID"]
#     Seq_ID2 = df.loc[i-1, "Seq ID"]
#     storage = []
    
#     if Seq_ID == Seq_ID2:
#         counter += 1
        
#     if Seq_ID != Seq_ID2:
#         for j in range(0,counter):
#             pass
            

# # Grouping Frames by Sequence Length - SLIDING WINDOW
# sequences=[]
# for i in range(NumberOfGroups):
#     CurrentGroup = []
#     for j in range(seq_len):
#         CurrentGroup.append(FrameNumbers[i+j])
#     sequences.append(CurrentGroup)
# # print(sequences)







# Finish Grouping Frames by Sequence ID into 1 big array
# Figure out how to detect when seq id changes and append frames attached to sequence id into array
# append that array into a big array

# Start working on mega loop 














# for i in range(1, count_rows - 1):
#     Seq_ID = df.loc[i, "Seq ID"]
#     Seq_ID2 = df.loc[i-1, "Seq ID"]

#     if Seq_ID != Seq_ID2:
#         NumberOfSequences += 1



     
     
        
     
"""
NumOfFrames = 10
time1 = list(range(1,NumOfFrames + 1))
# time2 = list(range(len(time1) + 1, len(time1) + NumOfFrames + 1))
# Ignore for now, this will be used to implement transition into next series
""" 
     
        
     
"""
fine_seq_id = [1]
counter = 1
for i in range(1, count_rows):
    if ((i % seq_len) != 0):
        fine_seq_id.append(counter)
    if ((i % seq_len) == 0):
        counter = counter + 1
        fine_seq_id.append(counter)

# Lastly add a column to the dataframe called Fine Seq ID
df['Fine Seq ID'] = fine_seq_id
print('\n\nNew row added to DataFrame\n--------------------------')
print(df)
"""

















"""
sequence_len = 6
NumberOfSequences = len(time1) - (sequence_len - 1)
sequences = []

# Grouping Frames by Sequence Length
for i in range(NumberOfSequences):
    CurrentSequence = []
    for j in range(sequence_len):
        sequences.append(time1[i+j])    
# print(sequences)


# Labels for Sequence ID
Seq_ID = []
Seq_ID_Length = len(sequences) * sequence_len

for i in range(NumberOfSequences):
    for j in range(sequence_len):
        Seq_ID.append(i+1)      
# print(Seq_ID)

Group = []
for i in range(len(Seq_ID)):
    CurrentGroup = []
    CurrentGroup.append(sequences[i])
    CurrentGroup.append(Seq_ID[i])
    Group.append(CurrentGroup)




df = pd.DataFrame(Group, columns=['Frame Number', 'Fine Seq ID'])
print(df)




# df.to_csv("JASH.csv")

# array = df[["Frame"]].to_numpy()

# print(array.tolist())

"""