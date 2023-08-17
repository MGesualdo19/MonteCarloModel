'''Michael Gesualdo August 2023
The following code uses the Geometric Brownian Motion (GBM) model
to predict the random price movements (change in price) of a stock based on the previous
price movement

The following formula is the basis for the model:
ΔS = S * (μΔt + σϵ(Δt)^(1/2))
Where...
S = the stock price
ΔS = the change in stock price
μ = the drift rate
σ = the standard deviation (volatility)
ϵ = the random variable
Δt = the elapsed time period
'''

import numpy as np
import matplotlib.pyplot as plot


class Equity:
    def __init__(self, name, initialPrice, volatility, histPrices, driftRate):
        self.name = name
        self.initialPrice = initialPrice
        self.volatility = volatility
        self.histPrices = histPrices
        self.driftRate = driftRate


def displayCurrentResult(result, name):
    # calculate mean and median prices of the simulations results
    meanPrices = [currentSimulation[-1] for currentSimulation in result]
    meanPricesF = np.mean(meanPrices)
    medianPricesF = np.median(meanPrices)

    # calculate percentiles to show price values of different ranges through the results
    percent = np.percentile(meanPrices, [10, 25, 50, 75, 90])
    stdDev = np.std(meanPrices)

    print("Current Equity: " + name)
    print("Mean final price:", meanPricesF)
    print("Median final price:", medianPricesF)
    print("Percentiles (10th, 25th, 50th, 75th, 90th):", percent)
    print("Standard deviation:", stdDev)
    print("The following is a line plot showing all completed simulations and their price trajectories.\n")

    # required information to produce line plot showing simulations
    time_steps = list(range(len(result[0])))
    plot.figure(figsize=(12, 6))
    for currentSimulation in result:
        plot.plot(time_steps, currentSimulation, linewidth=0.4)

    plot.xlabel("Time (Days)")
    plot.ylabel("Stock Price ($CAD)")
    plot.title("Monte Carlo (GBM) Simulation Results")
    plot.grid(True)
    plot.show()

    # required information for graph showing probabilities of outcomes
    # to achieve this a Cumulative Distribution Function (CDF) will be used
    # calculate CDF for the final prices by flattening and sorting/arranging data
    flatResult = np.array(result).flatten()
    sortedResult = np.sort(flatResult)
    cdf = np.arange(1, len(sortedResult) + 1) / len(sortedResult)

    print("The following is a CDF graph showing the probability of a given price outcome occurring over 252 days.\n")

    # input information to produce CDF graph
    plot.figure(figsize=(12, 6))
    plot.plot(sortedResult, cdf, linewidth=1.5)
    plot.xlabel("Stock Price")
    plot.ylabel("CDF (probability of a given price outcome occuring) ")
    plot.title("Cumulative Distribution Function (CDF) of Monte Carlo Simulation Results")
    plot.grid(True)
    plot.show()


def runSimulation(equities, numSimulations, timePeriod):
    # loop through different equities in the main
    for equity in equities:
        # create an array to store results
        result = []

        # execute all simulations by looping through
        for _ in range(numSimulations):
            # track the price in an array over time
            priceTrajectory = [equity.initialPrice]

            # loop through individual simulations time period
            for _ in range(0, timePeriod):
                # calculated random value
                random = np.random.normal(0, 1)

                # calculate GBM equation assuming Δt is 1 (i.e. 1 day)
                deltaS = priceTrajectory[-1] * (equity.driftRate + equity.volatility * random)
                # add new price to price array
                nextPrice = priceTrajectory[-1] + deltaS
                priceTrajectory.append(nextPrice)

            # add new trajectory to result array
            result.append(priceTrajectory)

        # call the method to display the results of the current equity
        displayCurrentResult(result, equity.name)


def fileRead(filePath):
    with open(filePath, "r") as file:
        # Read lines from the file and convert them to numbers
        lines = file.readlines()
        priceHist = [float(line.strip()) for line in lines]
        return priceHist

# natural log used to turn absolute price changes into relative returns
# the .diff determines the change in relative returns between 2 points
def volCalc(priceHist):
    # .std finds the standard deviation of all the changes in relative returns
    stockVol = np.std(np.diff(np.log(priceHist)))
    return stockVol


def driftCalc(priceHist):
    # .mean find the average value of the change in relative returns
    stockDrift = np.mean(np.diff(np.log(priceHist)))
    return stockDrift


if __name__ == "__main__":  # define the main method to create objects and run the simulation
    # read price values from text file applePriceHistory.txt
    applePath = "applePriceHistory"
    appleHist = fileRead(applePath)

    nfiPath = "nfiPriceHistory"
    nfiHist = fileRead(nfiPath)

    leonsPath = "leonsPriceHistory"
    leonsHist = fileRead(leonsPath)

    # calculate volatility (std dev) based on historical log returns
    appleVol = volCalc(appleHist)
    nfiVol = volCalc(nfiHist)
    leonsVol = volCalc(leonsHist)

    # calculate drift rate
    appleDrift = driftCalc(appleHist)
    nfiDrift = driftCalc(nfiHist)
    leonsDrift = driftCalc(leonsHist)

    # define Equity objects and simulation variables
    apple = Equity(name="Apple", initialPrice=appleHist[-1], volatility=appleVol, histPrices=appleHist,
                   driftRate=appleDrift)
    nfi = Equity(name="NFI", initialPrice=nfiHist[-1], volatility=nfiVol, histPrices=nfiHist, driftRate=nfiDrift)
    leons = Equity(name="Leons", initialPrice=leonsHist[-1], volatility=leonsVol, histPrices=leonsHist, driftRate=leonsDrift)

    numSimulations = 5000  # 5000 simulation iterations to be completed
    timePeriod = 252  # 252 days of trading (1 year in market)

    # run the simulation by calling the run function
    runSimulation([apple, nfi, leons], numSimulations, timePeriod)
