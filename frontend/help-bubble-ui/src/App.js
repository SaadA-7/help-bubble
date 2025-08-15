import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import styled from 'styled-components';

const ChatContainer = styled.div`
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Arial', sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 30px;
  color: white;
  
  h1 {
    font-size: 2.5rem;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
  }
  
  p {
    font-size: 1.1rem;
    margin: 10px 0 0 0;
    opacity: 0.9;
  }
`;

const ChatWindow = styled.div`
  background: white;
  border-radius: 15px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  overflow: hidden;
  height: 500px;
  display: flex;
  flex-direction: column;
`;

const MessagesContainer = styled.div`
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  background: #f8f9fa;
`;

const Message = styled.div`
  margin-bottom: 15px;
  display: flex;
  align-items: flex-start;
  ${props => props.isUser ? 'flex-direction: row-reverse;' : ''}
`;

const MessageBubble = styled.div`
  max-width: 70%;
  padding: 12px 16px;
  border-radius: 18px;
  ${props => props.isUser ? `
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin-left: 10px;
  ` : `
    background: #e9ecef;
    color: #333;
    margin-right: 10px;
  `}
  
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
`;

const MessageInfo = styled.div`
  font-size: 0.8rem;
  color: #666;
  margin-top: 5px;
  ${props => props.isUser ? 'text-align: right;' : 'text-align: left;'}
`;

const InputContainer = styled.div`
  padding: 20px;
  background: white;
  border-top: 1px solid #e9ecef;
`;

const InputForm = styled.form`
  display: flex;
  gap: 10px;
`;

const QuestionInput = styled.input`
  flex: 1;
  padding: 12px 16px;
  border: 2px solid #e9ecef;
  border-radius: 25px;
  font-size: 1rem;
  outline: none;
  
  &:focus {
    border-color: #667eea;
  }
`;

const SendButton = styled.button`
  padding: 12px 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 25px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: bold;
  transition: transform 0.2s;
  
  &:hover {
    transform: translateY(-2px);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const LoadingIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  color: #666;
  font-style: italic;
`;

const LoadingDots = styled.div`
  display: inline-block;
  position: relative;
  width: 40px;
  height: 10px;
  
  div {
    position: absolute;
    top: 0;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #667eea;
    animation-timing-function: cubic-bezier(0, 1, 1, 0);
  }
  
  div:nth-child(1) {
    left: 8px;
    animation: loading1 0.6s infinite;
  }
  
  div:nth-child(2) {
    left: 8px;
    animation: loading2 0.6s infinite;
  }
  
  div:nth-child(3) {
    left: 32px;
    animation: loading2 0.6s infinite;
  }
  
  div:nth-child(4) {
    left: 56px;
    animation: loading3 0.6s infinite;
  }
  
  @keyframes loading1 {
    0% { transform: scale(0); }
    100% { transform: scale(1); }
  }
  
  @keyframes loading3 {
    0% { transform: scale(1); }
    100% { transform: scale(0); }
  }
  
  @keyframes loading2 {
    0% { transform: translate(0, 0); }
    100% { transform: translate(24px, 0); }
  }
`;

const ExampleQuestions = styled.div`
  margin-top: 15px;
  
  h4 {
    margin: 0 0 10px 0;
    color: #666;
    font-size: 0.9rem;
  }
`;

const ExampleButton = styled.button`
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 15px;
  padding: 8px 12px;
  margin: 5px 5px 5px 0;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s;
  
  &:hover {
    background: #e9ecef;
    border-color: #667eea;
  }
`;

const StatusIndicator = styled.div`
  position: fixed;
  top: 20px;
  right: 20px;
  padding: 8px 12px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: bold;
  ${props => props.connected ? `
    background: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
  ` : `
    background: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
  `}
`;

function App() {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const messagesEndRef = useRef(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const exampleQuestions = [
    "How long do I have to return an item?",
    "What are your shipping options?",
    "Do you accept PayPal?",
    "What warranty comes with electronics?",
    "How do I track my order?",
    "Do you offer student discounts?"
  ];

  useEffect(() => {
    // Check API health on component mount
    checkApiHealth();
    
    // Add welcome message
    setMessages([{
      id: Date.now(),
      text: "Hello! I'm HelpBubble, your customer support assistant. How can I help you today?",
      isUser: false,
      timestamp: new Date().toLocaleTimeString(),
      confidence: 1.0
    }]);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`, { timeout: 5000 });
      setConnected(response.data.status === 'healthy');
    } catch (error) {
      console.error('API health check failed:', error);
      setConnected(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || loading) return;

    const userMessage = {
      id: Date.now(),
      text: question,
      isUser: true,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    
    const currentQuestion = question;
    setQuestion('');

    try {
      const response = await axios.post(`${API_BASE_URL}/ask`, {
        question: currentQuestion,
        user_id: `user_${Date.now()}`
      }, {
        timeout: 30000
      });

      const botMessage = {
        id: Date.now() + 1,
        text: response.data.answer,
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
        confidence: response.data.confidence,
        responseTime: response.data.response_time,
        contextUsed: response.data.context_used
      };

      setMessages(prev => [...prev, botMessage]);
      setConnected(true);
      
    } catch (error) {
      console.error('Error asking question:', error);
      
      const errorMessage = {
        id: Date.now() + 1,
        text: error.response?.status === 500 
          ? "I'm having trouble processing your request right now. Please try again or contact support."
          : "I couldn't connect to the support system. Please check your internet connection and try again.",
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
        isError: true
      };

      setMessages(prev => [...prev, errorMessage]);
      setConnected(false);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (exampleQuestion) => {
    setQuestion(exampleQuestion);
  };

  return (
    <ChatContainer>
      <StatusIndicator connected={connected}>
        {connected ? '🟢 Connected' : '🔴 Disconnected'}
      </StatusIndicator>
      
      <Header>
        <h1>🛍️ HelpBubble</h1>
        <p>AI-Powered Customer Support Assistant</p>
      </Header>

      <ChatWindow>
        <MessagesContainer>
          {messages.map((message) => (
            <Message key={message.id} isUser={message.isUser}>
              <MessageBubble isUser={message.isUser} isError={message.isError}>
                {message.text}
                <MessageInfo isUser={message.isUser}>
                  {message.timestamp}
                  {message.confidence && !message.isUser && (
                    <span> • Confidence: {(message.confidence * 100).toFixed(1)}%</span>
                  )}
                  {message.responseTime && (
                    <span> • {(message.responseTime * 1000).toFixed(0)}ms</span>
                  )}
                </MessageInfo>
              </MessageBubble>
            </Message>
          ))}
          
          {loading && (
            <Message isUser={false}>
              <MessageBubble isUser={false}>
                <LoadingIndicator>
                  Thinking
                  <LoadingDots>
                    <div></div>
                    <div></div>
                    <div></div>
                    <div></div>
                  </LoadingDots>
                </LoadingIndicator>
              </MessageBubble>
            </Message>
          )}
          
          <div ref={messagesEndRef} />
        </MessagesContainer>

        <InputContainer>
          <InputForm onSubmit={handleSubmit}>
            <QuestionInput
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask me about returns, shipping, payments, warranties..."
              disabled={loading}
            />
            <SendButton type="submit" disabled={loading || !question.trim()}>
              {loading ? 'Sending...' : 'Send'}
            </SendButton>
          </InputForm>

          <ExampleQuestions>
            <h4>Try asking:</h4>
            {exampleQuestions.map((example, index) => (
              <ExampleButton
                key={index}
                onClick={() => handleExampleClick(example)}
                disabled={loading}
              >
                {example}
              </ExampleButton>
            ))}
          </ExampleQuestions>
        </InputContainer>
      </ChatWindow>
    </ChatContainer>
  );
}

export default App;