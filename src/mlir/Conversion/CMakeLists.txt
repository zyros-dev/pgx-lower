# Add subdirectories for each conversion pass
add_subdirectory(RelAlgToDB)
add_subdirectory(DBToDSA)
add_subdirectory(DSAToLLVM)

# Future conversion passes will be added here:
# add_subdirectory(DSAToStd)
# add_subdirectory(UtilToLLVM)