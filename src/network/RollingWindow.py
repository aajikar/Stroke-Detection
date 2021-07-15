def PatValues(data):
    '''
    Gets Patient ID values and number of patients
    
    Parameters
    ----------
    data : pandas dataframe
        Metadata.

    Returns
    -------
    Pat_ID_Values : list
        List with each Patient ID as entry.
    UniquePat : int
        Number of unique patients.

    '''
    Pat_ID_Values = data['Patient ID'].value_counts().sort_index()
    UniquePat = len(Pat_ID_Values) 
    
    return Pat_ID_Values, UniquePat


def PatSequences(data, UniquePat):
    '''
    Gets number of Seq ID per Patient ID and number of frames per Seq ID

    Parameters
    ----------
    data : pandas dataframe
         Metadata.
    UniquePat : int
        Number of unique patients.

    Returns
    -------
    SeqPerPat : list
        List with each entry corresponding to the number of Seq ID per patient.
    UniqPatSeq : pandas.core.series.Series
        Number of frames grouped by Seq ID furthur grouped by Patient ID.
    '''
    UniqPatSeq = data.value_counts(['Patient ID', 'Seq ID']).sort_index()
    SeqPat = data.groupby("Patient ID")["Seq ID"].nunique()
    
    SeqPerPat = []
    for i in range(UniquePat):
        SeqPerPat.append(SeqPat.iloc[i])
    
    return SeqPerPat, UniqPatSeq


def FrameIndexingByPat(UniquePat, SeqPerPat, UniqPatSeq):
    '''
    Indexing tool to create list with indexes of all frames
    
    Parameters
    ----------
    UniquePat : int
        Number of unique patients.
    SeqPerPat : list
        List with each entry correspdonding to the number of Seq ID per patient.
    UniqPatSeq : pandas.core.series.Series
        Number of frames grouped by Seq ID furthur grouped by Patient ID.

    Returns
    -------
    IndexByPatient : list
        List with index of frame numbers grouped by Seq ID, and furthur, Patient ID.
    '''
    
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


def NumSequences(UniquePat, SeqLen, IndexByPatient, SeqPerPat):
    '''
    Gets Number of possible sequences within each Seq ID and Patient ID

    Parameters
    ----------
    UniquePat : int
        Number of unique patients.
    SeqLen : int
        Sequence Length.
    IndexByPatient : list
        List with index of frame numbers grouped by Seq ID, and furthur, Patient ID.
    SeqPerPat : list
        List with each entry correspdonding to the number of Seq ID per patient.

    Returns
    -------
    NumOfSequences : list
        List with the number of possible samples per Seq ID per Patient ID.

    '''
    NumOfSequences = []
    for i in range (UniquePat):
        CurrentGroup = []
        for j in range(SeqPerPat[i]):
            Num = len(IndexByPatient[i][j]) - (SeqLen - 1)
            CurrentGroup.append(Num)
        NumOfSequences.append(CurrentGroup)  
        
    return NumOfSequences


def RollingWindow(UniquePat, SeqLen, NumOfSequences, IndexByPatient):
    '''
    Creates rolling window

    Parameters
    ----------
    UniquePat : int
        Number of unique patients.
    SeqLen : int
        Sequence Length.
    NumOfSequences : list
        List with each entry correspdonding to the number of Seq ID per patient.
    IndexByPatient : list
        List with index of frame numbers grouped by Seq ID, and furthur, Patient ID.

    Returns
    -------
    IndexedWindow : list
        List of all samples with index numbers.

    '''
    IndexedWindow = []
    for i in range(UniquePat):
        for j in range(len(NumOfSequences[i])):
            for k in range(NumOfSequences[i][j]):
                IndexedWindow.append(IndexByPatient[i][j][k:k+SeqLen])
                
    return IndexedWindow


def CreateRollingWindow(data, SeqLen):
    '''
    Main function that calls on all other functions to create rolling window

    Parameters
    ----------
    data : pandas dataframe
        Metadata.
    SeqLen : int
        Sequence Length.

    Returns
    -------
    IndexedRW : list
        List of all samples with index numbers.

    '''

    PatVals, NumPat = PatValues(data)

    SeqPerPat, UniqPatSeq = PatSequences(data, NumPat)
    
    IndexedPatients = FrameIndexingByPat(NumPat, SeqPerPat, UniqPatSeq)

    NumOfSequences = NumSequences(NumPat, SeqLen, IndexedPatients, SeqPerPat)
   
    IndexedRW = RollingWindow(NumPat, SeqLen, NumOfSequences, IndexedPatients)     

    return IndexedRW
    
    
    
    



