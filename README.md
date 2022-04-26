# Modeling Inter-Regional COVID Dynamics and Transmission  

By: Skyler Wu, Alex Rojas, Oliver Cheng

_Final project for AM216_

*Primary Database:* [Covid Open Data](https://health.google.com/covid-19/open-data/raw-data)

*Problem:* The next surge of cases is particularly difficult to predict. How far ahead can we predict?

*Goal:* can we accurately forecast COVID-19 cases 1, 2, or 3+ months ahead?

*Starting Intuitions:*
1. If cities A and B are very interconnected, then COVID cases in city A will influence COVID cases in city B → NETWORK!
2. Landmark events like vaccination, mask mandates / travel restrictions, and large gatherings (Thanksgiving travel, Spring break) → outsized influence on cases → need to retrain weights periodically.

Methods + Steps:
1. **Graph Neural Network:** each state/city is a node, and edges are differential equations / other features that model infectivity, transmission, trade/interconnectedness, etc.
2. **Recurrent Neural Network:** take observed case counts at each time t as input features per timestep → splice into feed-forward network for forecasting.
3. **Vector Autoregression:** Model time-series data using linear transformations.
4. **Mechanistic Models:** SIR, SIR-SEIS, and their variations fitted to real data using JAX.

