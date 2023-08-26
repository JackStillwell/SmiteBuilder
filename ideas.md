# Improved "Historical" Builder

* Filter out all instances where Role doesn't match -- if no God + Role match is available, look into substituting a similar God if possible
* Add additional score for builds which considers the allied and opposing gods in a match -- probably just a BNB that goes through and ranks the likelihood of seeing that item built. 

# New Path / Stat-based Builder

* Need to translate stats into "passive" and then in-game results based upon stats
* When items are changed, do a grep of their passive for number changes and apply a modifier, or if unknown just set it to 0 and re-evaluate later with better information.
* Use these item "values" to craft the best builds for gods, after figuring out which combinations of values benefit gods most at the end of their games. Will want to consider game ending times and the differences in builds between them.
* Because time is a factor, will likely want to give a build order.