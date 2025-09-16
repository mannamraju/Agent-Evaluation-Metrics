# Documentation

This folder contains comprehensive documentation for the CXR Report Metrics evaluation system.

## 📚 Documentation Files

### Setup Guides
- **`SEMANTIC_EMBEDDING_SETUP.md`** - CheXbert model setup and configuration instructions

### Implementation History
- **`IMPLEMENTATION_HISTORY.md`** - Comprehensive history of all metric implementations, test results, and architectural evolution

## 📖 Additional Documentation Locations

### Metric-Specific Documentation
Each metric has its own documentation within its folder:
- `CXRMetric/metrics/bleu/README.md` - BLEU metrics documentation
- `CXRMetric/metrics/rouge/README.md` - ROUGE metrics documentation  
- `CXRMetric/metrics/perplexity/README.md` - Perplexity metrics documentation
- `CXRMetric/metrics/composite/README.md` - Composite metrics documentation
- `CXRMetric/metrics/semantic_embedding/README.md` - Semantic embedding documentation

### Testing Documentation
- `CXRMetric/metrics/METRICS_TESTING_SUMMARY.md` - Comprehensive testing infrastructure overview

### Utility Documentation
- `utility_scripts/README.md` - General utility scripts documentation
- `azure_deployment/README.md` - Azure deployment guides and scripts

## 🎯 Quick Reference

### For Users
1. **Getting Started**: See main `README.md` in project root
2. **Setup Requirements**: Check `SEMANTIC_EMBEDDING_SETUP.md` for CheXbert setup
3. **Running Evaluations**: Use `utility_scripts/test_metric.py` or the Jupyter notebook

### For Developers
1. **Architecture Overview**: Read `IMPLEMENTATION_HISTORY.md` for evolution details
2. **Metric Development**: See `CXRMetric/metrics/base_evaluator.py` for the interface
3. **Testing Patterns**: Check `METRICS_TESTING_SUMMARY.md` for testing infrastructure

### For Deployment
1. **Azure Setup**: See `azure_deployment/` folder for comprehensive guides
2. **GPU Configuration**: Check `IMPLEMENTATION_HISTORY.md` for hardware requirements
3. **Performance Optimization**: Review individual metric documentation for tuning options
