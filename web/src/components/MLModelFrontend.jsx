import React, { useState, useEffect } from 'react';
import { Play, RefreshCw, Database, TrendingUp, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import { apiService } from '../services/api';
import IrisForm from './IrisForm';
import CancerForm from './CancerForm';
import ResultsDisplay from './ResultsDisplay';
import ModelTraining from './ModelTraining';

const MLModelFrontend = () => {
  const [selectedDataset, setSelectedDataset] = useState('iris');
  const [predictions, setPredictions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');
  const [modelStatus, setModelStatus] = useState({});

  useEffect(() => {
    checkApiHealth();
    // Poll for model status every 2 seconds
    const interval = setInterval(checkModelStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const checkApiHealth = async () => {
    try {
      await apiService.getHealth();
      setApiStatus('healthy');
    } catch (error) {
      setApiStatus('error');
      toast.error('API connection failed');
    }
  };

  const checkModelStatus = async () => {
    try {
      const response = await apiService.request('ready/full');
      setModelStatus(response.model_status || {});

      // Update API status based on model readiness
      if (response.ready) {
        if (response.all_models_loaded) {
          setApiStatus('healthy');
        } else if (Object.values(response.model_status || {}).some(s => s === 'failed')) {
          setApiStatus('warning');
        } else {
          setApiStatus('loading');
        }
      } else {
        setApiStatus('error');
      }
    } catch (error) {
      console.error('Failed to check model status:', error);
      setApiStatus('error');
    }
  };

  const getStatusColor = () => {
    switch (apiStatus) {
      case 'healthy': return 'bg-green-100 text-green-800';
      case 'warning': return 'bg-orange-100 text-orange-800';
      case 'error': return 'bg-red-100 text-red-800';
      case 'loading': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusText = () => {
    switch (apiStatus) {
      case 'healthy': return 'All Models Ready';
      case 'warning': return 'Some Models Failed';
      case 'error': return 'API Error';
      case 'loading': return 'Models Loading...';
      default: return 'Checking...';
    }
  };

  const getStatusIcon = () => {
    switch (apiStatus) {
      case 'healthy': return 'bg-green-500';
      case 'warning': return 'bg-orange-500';
      case 'error': return 'bg-red-500';
      case 'loading': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const handlePredict = async (formData) => {
    setIsLoading(true);
    setPredictions(null);

    try {
      let response;
      if (selectedDataset === 'iris') {
        response = await apiService.predictIris(formData);
      } else {
        response = await apiService.predictCancer(formData);
      }

      setPredictions(response);
      toast.success('Prediction completed successfully!');
    } catch (error) {
      console.error('Prediction error:', error);
      toast.error(`Prediction failed: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTrainModel = async () => {
    setIsTraining(true);
    try {
      toast.success('Model training started! This may take a few minutes...');
      // In a real app, you'd have an endpoint for training
      // For now, we'll simulate training
      await new Promise(resolve => setTimeout(resolve, 3000));
      toast.success('Model training completed!');
    } catch (error) {
      toast.error(`Training failed: ${error.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const datasets = [
    {
      id: 'iris',
      name: 'Iris Classification',
      description: 'Classify iris flowers into species based on measurements',
      features: ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
      classes: ['Setosa', 'Versicolor', 'Virginica']
    },
    {
      id: 'cancer',
      name: 'Breast Cancer Diagnosis',
      description: 'Predict malignant vs benign breast cancer diagnosis',
      features: ['30 diagnostic features'],
      classes: ['Malignant', 'Benign']
    }
  ];

  const currentDataset = datasets.find(d => d.id === selectedDataset);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Machine Learning Models</h2>
            <p className="text-gray-600 mt-1">
              Select a dataset and make predictions using trained models
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${getStatusColor()}`}>
              <div className={`w-2 h-2 rounded-full ${getStatusIcon()}`}></div>
              <span>{getStatusText()}</span>
            </div>
            <button
              onClick={checkApiHealth}
              className="btn-outline btn-sm"
              disabled={apiStatus === 'checking'}
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </button>
          </div>
        </div>

        {/* Model Status Details */}
        {Object.keys(modelStatus).length > 0 && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(modelStatus).map(([model, status]) => (
              <div key={model} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                <span className="text-sm font-medium text-gray-700">
                  {model === 'iris_random_forest' ? 'Iris Model' : 'Cancer Model'}
                </span>
                <span className={`text-xs px-2 py-1 rounded-full ${
                  status === 'loaded' ? 'bg-green-100 text-green-800' :
                  status === 'training' ? 'bg-blue-100 text-blue-800' :
                  status === 'failed' ? 'bg-red-100 text-red-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {status}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Dataset Selection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {datasets.map((dataset) => (
          <div
            key={dataset.id}
            className={`card cursor-pointer transition-all duration-200 hover:shadow-md ${
              selectedDataset === dataset.id
                ? 'ring-2 ring-primary-500 border-primary-200'
                : 'hover:border-gray-300'
            }`}
            onClick={() => setSelectedDataset(dataset.id)}
          >
            <div className="flex items-start space-x-3">
              <div className={`p-2 rounded-lg ${
                selectedDataset === dataset.id
                  ? 'bg-primary-100 text-primary-600'
                  : 'bg-gray-100 text-gray-600'
              }`}>
                <Database className="h-5 w-5" />
              </div>
              <div className="flex-1">
                <h3 className="font-medium text-gray-900">{dataset.name}</h3>
                <p className="text-sm text-gray-600 mt-1">{dataset.description}</p>
                <div className="mt-2">
                  <div className="flex items-center text-xs text-gray-500">
                    <span className="mr-4">Features: {dataset.features.join(', ')}</span>
                  </div>
                  <div className="flex items-center text-xs text-gray-500 mt-1">
                    <span>Classes: {dataset.classes.join(', ')}</span>
                  </div>
                </div>
              </div>
              {selectedDataset === dataset.id && (
                <div className="text-primary-600">
                  <TrendingUp className="h-5 w-5" />
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Prediction Form */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">
              {currentDataset?.name} Prediction
            </h3>
            <p className="text-sm text-gray-600">
              Enter values to get a prediction from the trained model
            </p>
          </div>

          {selectedDataset === 'iris' ? (
            <IrisForm onPredict={handlePredict} isLoading={isLoading} />
          ) : (
            <CancerForm onPredict={handlePredict} isLoading={isLoading} />
          )}
        </div>

        {/* Results */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Prediction Results</h3>
            <p className="text-sm text-gray-600">
              Model predictions and confidence scores
            </p>
          </div>

          <ResultsDisplay 
            predictions={predictions} 
            dataset={selectedDataset}
            isLoading={isLoading}
          />
        </div>
      </div>

      {/* Model Training */}
      <ModelTraining 
        dataset={selectedDataset}
        onTrain={handleTrainModel}
        isTraining={isTraining}
        setIsTraining={setIsTraining}
      />
    </div>
  );
};

export default MLModelFrontend; 

