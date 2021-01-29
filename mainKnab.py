from portfolioActivity import *
from generalMethodsClasses import *

##Import file

##Define which folders to go in
rawDataCollection = "rawData"
newDataCollection = "newData"
# data = importPortfolioActivity(dataTypeConvertAll= datatypeConvertAll, columsToParse)

##SAMPLE OPERATIONS
changeToNewData()
pat = pd.read_csv("total_portfolio_activity_larger_sample.csv")
# pat = importChunk("total_portfolio_activity_larger.csv", chunksize= 250000)
changeToProject()


pass