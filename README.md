# Nuclear Reactor AI Control System
## Anomaly & Fault Detection with Machine Learning

A comprehensive AI-powered nuclear reactor control and monitoring system using Deep Reinforcement Learning (Soft Actor-Critic) models.


## 🎯 Project Overview

This project demonstrates an advanced control system for nuclear reactors using pre-trained SAC (Soft Actor-Critic) reinforcement learning models. The system provides:

- **Real-time Reactor Monitoring**: Live dashboards with gauges, graphs, and metrics
- **AI-Powered Control**: Automatic reactor control using trained SAC agents
- **Manual Override**: User-adjustable controls for testing and education
- **Scenario Testing**: Multiple test scenarios (LOFA, rod malfunction, power ramp)
- **Event Detection**: Anomaly detection and critical event logging
- **Performance Metrics**: Comprehensive statistics on model performance

---


## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+
- Ports 3000 (frontend) and 8000 (backend) available

### Installation & Run

```bash
# Terminal 1: Start Backend
cd backend
pip install -r requirements.txt
python run.py
# Expected: "Running on http://localhost:8000"

# Terminal 2: Start Frontend
cd frontend
npm install
npm run dev
# Expected: "Ready on http://localhost:3000"
```

### First Simulation

1. Open [http://localhost:3000](http://localhost:3000)
2. Select a model from dropdown (e.g., "Enhanced SAC Agent")
3. Select a scenario (e.g., "Normal Operation")
4. Click "Start Simulation"
5. Watch the AI control the reactor in real-time!

---

## 📊 Dashboard Features
<img width="1622" height="2347" alt="lofa full" src="https://github.com/user-attachments/assets/86f38d71-c3a4-4731-ad2d-861ab6f19b4e" />

### Left Panel - Reactor Visualization
- **4 Circular Gauges**: Power, Fuel Temperature, Coolant Temperature, Pressure
- **Control Rods Display**: Visual representation of power and precursor levels
- **Temperature Heatmap**: Color-coded thermal profile

### Center Panel - Simulation Control
- **Model Selector**: Choose between Enhanced or Optimized SAC agent
- **Scenario Selector**: Test different reactor conditions
- **Control Buttons**: Start, Stop, Pause, Reset
- **Manual Controls**: Adjustable sliders for control rods and coolant flow
- **Status Card**: Current simulation state and progress

### Right Panel - Metrics & Analysis
- **Event Log**: Timestamped events (100 max, circular buffer)
- **Metrics Summary**: Statistics (reward, steps, temperature peaks, etc.)
- **Score Cards**: Key performance indicators
- **Real-time Graphs**: Power, temperatures, and pressure trends

<img width="1636" height="2210" alt="completed dashboard" src="https://github.com/user-attachments/assets/a505e84e-b016-4593-8461-08a59c242ed1" />

---

## 🔌 API Endpoints

### Health & Status
```
GET  /api/health           → Backend health check
GET  /api/status           → System status & available models/scenarios
```

### Models
```
GET  /api/models           → List all models
GET  /api/models/{id}      → Model details
POST /api/models/{id}/load → Load model into memory
```

### Scenarios
```
GET  /api/scenarios        → List available scenarios
```

### Simulation Control
```
POST /api/simulation/reset     → Reset environment
POST /api/simulation/start     → Start with model & scenario
POST /api/simulation/step      → Execute AI step
POST /api/simulation/action    → Execute manual action
GET  /api/simulation/state     → Get current state
POST /api/simulation/stop      → Stop & get summary
```


---

## 🧠 Models Included

### Enhanced SAC Agent
- **Training Steps**: 250,000
- **Average Reward**: 48.6 points/step
- **Network Size**: Large
- **Performance**: Excellent control stability
- **Use Case**: Production control
- **Location**: `python/SAC_enhanced_model/nuclear_reactor_sac/models/enhanced/best_model.pth`
<img width="2882" height="2147" alt="best_episode" src="https://github.com/user-attachments/assets/e6e17aa2-4491-4b9d-bb41-e4af425efc64" />

### Optimized SAC Agent
- **Training Steps**: 150,000
- **Average Reward**: 7.3 points/step  
- **Network Size**: Smaller
- **Performance**: Good control with faster inference
- **Use Case**: Real-time edge deployment
- **Location**: `python/SAC_model/models/optimized/best_model.pth`

---

## 🧪 Test Scenarios

### Normal Operation
Default safe reactor operation at nominal 100 MW. Used for baseline testing.

### LOFA (Loss of Coolant Flow Accident)
Simulates loss of coolant flow (40% reduction) starting at t=5s. Tests AI's ability to manage reactor without added cooling.

### Rod Malfunction
Control rod stuck at 50% insertion starting at t=3s. Tests AI's ability to control power with limited rod movement.

### Power Ramp
Gradual demand increase to 120 MW. Tests safe power escalation and thermal management.

---

## 🎛️ User Controls

### Automatic Mode
- Select model and scenario
- Click "Start"
- Watch AI maintain reactor stability
- Monitor real-time metrics and events

### Manual Mode
- Start simulation in any scenario
- Adjust control rods (-1.0 fully retracted, 1.0 fully inserted)
- Adjust coolant flow (-1.0 decrease, 1.0 increase)
- Compare your manual control against the AI

### Analysis Mode
- Stop simulation at any time
- View final metrics and performance
- Compare multiple models on same scenario
- Export event history and metrics


## ✨ Features Implemented

### Completed ✅
- ✅ Production-ready Flask backend
- ✅ Complete Next.js frontend
- ✅ 20+ React components
- ✅ Real-time reactor monitoring
- ✅ AI model integration
- ✅ Manual control override
- ✅ Multiple scenarios
- ✅ Performance metrics
- ✅ Event logging
- ✅ Comprehensive documentation
- ✅ Type-safe code
- ✅ Error handling
- ✅ CORS configuration

### Future Enhancements 🔮
- WebSocket real-time updates
- Model comparison mode
- Historical data storage
- Advanced charting (Recharts)
- Dark mode
- Export capabilities
- Model training UI
- Docker containerization



---

## 📝 License

This project is licensed under the terms specified in the MIT License.

---


---

## 🎯 Summary

A **complete, production-ready AI-powered nuclear reactor control system** demonstrating:
- ✅ Advanced AI control (Soft Actor-Critic)
- ✅ Real-time monitoring and visualization
- ✅ Comprehensive documentation
- ✅ Professional code quality
- ✅ Modern web technologies
- ✅ Ready to run locally
- ✅ Ready to deploy

**Status**: ✅ **FULLY FUNCTIONAL**

---

## 🚀 Get Started Now!

```bash
# 1. Backend (Terminal 1)
cd backend && pip install -r requirements.txt && python run.py

# 2. Frontend (Terminal 2)
cd frontend && npm install && npm run dev

# 3. Open Browser
http://localhost:3000
```

**Then explore, experiment, and enjoy controlling a nuclear reactor with AI!** 🔋⚛️

---

**Version**: 1.0.0  
**Last Updated**: April 2025  
**Status**: Production Ready
