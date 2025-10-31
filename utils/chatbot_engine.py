eng
import re
import random
import json
from datetime import datetime

class EduChatBot:
    def __init__(self):
        self.responses = {
            'greeting': [
                "Hello! I'm your Student Engagement Assistant. How can I help you today?",
                "Hi there! I'm here to help with student engagement questions.",
                "Welcome! I can assist you with student risk analysis and engagement strategies."
            ],
            'risk_analysis': [
                "I can help analyze student risk factors. The system considers attendance, GPA, behavioral metrics, and socio-economic factors.",
                "Risk analysis is based on multiple factors including academic performance, engagement levels, and personal circumstances.",
                "Our AI model predicts engagement risk using machine learning on academic and behavioral data."
            ],
            'intervention': [
                "For high-risk students, I recommend: 1) Academic counseling 2) Mentorship programs 3) Regular check-ins",
                "Effective interventions include personalized learning plans, peer mentoring, and early alert systems.",
                "Consider these interventions: academic support, mental health resources, and career guidance."
            ],
            'attendance': [
                "Attendance below 75% significantly increases disengagement risk. Monitor patterns and reach out early.",
                "Poor attendance often correlates with other risk factors. Look for weekly patterns.",
                "Regular attendance is a strong predictor of student success and engagement."
            ],
            'gpa': [
                "Students with GPA below 2.5 are at higher risk. Offer tutoring and academic support.",
                "Monitor GPA trends - declining grades often indicate emerging issues.",
                "Academic performance is a key indicator. Look for semester-to-semester changes."
            ],
            'prediction': [
                "Our prediction model uses Random Forest algorithm with 85% accuracy on test data.",
                "Predictions are based on historical data and real-time behavioral metrics.",
                "The AI model continuously learns from new data to improve risk assessment accuracy."
            ],
            'fallback': [
                "I'm still learning about student engagement. Could you rephrase your question?",
                "I specialize in student risk analysis. Ask me about interventions, predictions, or risk factors.",
                "I'm designed to help with student engagement topics. Try asking about risk factors or interventions."
            ]
        }
        
        self.patterns = {
            'greeting': [r'hello', r'hi', r'hey', r'good morning', r'good afternoon'],
            'risk_analysis': [r'risk', r'predict', r'analysis', r'factors?', r'engagement'],
            'intervention': [r'intervention', r'help', r'support', r'what to do', r'action'],
            'attendance': [r'attendance', r'present', r'absent', r'class'],
            'gpa': [r'gpa', r'grade', r'marks?', r'performance'],
            'prediction': [r'prediction', r'model', r'ai', r'machine learning', r'algorithm']
        }
    
    def classify_intent(self, message):
        message = message.lower().strip()
        
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, message):
                    return intent
        
        return 'fallback'
    
    def get_response(self, message, student_data=None):
        intent = self.classify_intent(message)
        responses = self.responses.get(intent, self.responses['fallback'])
        
        # Select random response from the category
        response = random.choice(responses)
        
        # Add student-specific context if available
        if student_data and intent in ['risk_analysis', 'intervention']:
            if student_data.get('risk_level') == 'High':
                response += " This student shows high risk indicators and needs immediate attention."
            elif student_data.get('risk_level') == 'Medium':
                response += " This student shows moderate risk - regular monitoring is recommended."
        
        return {
            'response': response,
            'intent': intent,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_suggested_questions(self):
        return [
            "How does the risk prediction work?",
            "What interventions help high-risk students?",
            "How important is attendance for engagement?",
            "Tell me about GPA risk factors",
            "What data does the AI model use?"
        ]

# Global chatbot instance
chatbot = EduChatBot()