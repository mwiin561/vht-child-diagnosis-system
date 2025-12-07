---

## 🏗️ System Architecture

The architecture of this cognitive system follows a **four-layer pipeline**, ensuring clear modularity, interpretability, and clinical relevance.

### **🔹 1. NLP Processing Layer**
- Accepts free-text symptoms in **Luganda or English**
- Cleans and normalizes text
- Detects keywords using a curated medical vocabulary
- Performs negation handling (“sita”, “siko”, “not”, “no”)
- Extracts metadata: **age** and **duration of illness**

### **🔹 2. Machine Learning Layer**
- Takes extracted symptom indicators as input  
- Random Forest model trained on a curated childhood disease dataset  
- Outputs probabilities for:
  - Malaria  
  - Pneumonia  
  - Diarrhea  

### **🔹 3. Knowledge Graph Reasoning Layer**
- Encodes IMCI-style medical associations  
- Rules link symptoms → diseases with weights  
- Handles danger signs (convulsions, lethargy, chest-indrawing)  
- Produces symbolic scores and interpretable rule traces  

### **🔹 4. Hybrid Reasoning Layer**
Combines both approaches:

Final Score = 0.7 × ML_probability + 0.3 × KG_reasoning_score



Provides:
- Final hybrid diagnosis  
- Risk level (Low / Moderate / High)  
- Fired rules for explainability  

### **🔹 5. Streamlit User Interface**
- Simple input box for symptoms  
- Sidebar with ready-made test cases  
- Displays:
  - Diagnosis  
  - Risk  
  - Symptoms detected  
  - ML probabilities  
  - Fired KG rules  
  - Danger signs  

### 📌 Architecture Diagram

*(Place your PNG diagram here in GitHub using the "Upload file" button.)*

Example Markdown:


![System Architecture](Architecture_diagram.png)
📄 Requirements
The project requires Python libraries for:

NLP text cleaning

Machine learning inference

Knowledge graph reasoning

Streamlit UI deployment

Cloudflare tunneling (for Colab deployment)



