1) BindingDB.py
	Renamed columns
	Cleaned interaction column values
	Taking UniProt IDs as SwissProt IDs. Any NaN in Swissprot IDs is filled with TrEMBL Primary IDs
	Removing interactions for which Uniprot ID , SMILES , INCHI , INCHI Key or PCID is NaN
	Separating dataset into Target , Ligand & Interact dataset
	Some UniProt IDs are obsolete. [In the Dataset, SwissPROT & TrEMBL name is NaN][Not perfect].
		Removed such entries from Target & Interact dataset

2) fix_bdb_targets.py
	Created 2 target files
		One file contains GENE IDs as UniprotID::MutationID
		The other contains  IDs as UniprotID
		[MutationID here refer to mutations (A997L) or regions of interest(2-33)]
		[Some MutationID also contains unwanted data that cannot be considered such as names]
		[But such MutationIDs also have the above useful data after a visual check]
		[But there is a chance I might have missed some]
		
3) fix_bdb_ligands.py
	Created LIG_ID
	2 copies of same Ligand dataset, one for Targets with mutation & the other for Targets without mutation
	Removed ligands with same PCID

4) fix_bdb_interacts1.py
	Created 2 interact files; with & without mutation data
	Removed interactions for ligands with same PCID
	Matching GENE & LIG IDs to Uniprot ID & INCHI Key

5) separate_interacts.py
	For both types of interact data
	Interact separated based on KI, KD, IC50, EC50, KON & Koff
	Interact values of type "<>x" & "NV," kept as NaN.
		Original values in ['REF ] column
		
6) fix_bdb_interacts2.py
	Removed interactions for which Patent/DOI and PAID is same
		DOIs referring to PATENT will have value in both columns
		Therefore, unless specified, no DOI does not refer to some patent
		DOIs are not preprints. Therefore, each DOI is unique
	Removed interactions for which DOI & PATENT is NaN
	Interactions with same Patent/DOI but different PAID kept
	
	For an interaction, duplicates are present which have same reference & value but differ only by the presence/absence of PAID
	Int1	10	Ref1	PAID1
	Int1	10	Ref1	NaN
	In such cases, it is unknown whether the interaction has the same PAID and is a duplicate to be removed or if it has different PAID
	& should be kept. Only such interactions with NaN PAID removed.
	Singular interactions with NaN PAID are kept
	
7) create_bdb_matrix.py
	Matrix created with values <= 10000nM, and duplicates's median found