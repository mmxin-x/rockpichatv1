// src/ErrorBoundary.jsx
import React from 'react';

export default class ErrorBoundary extends React.Component {
  state = { hasError: false };

  static getDerivedStateFromError() { 
    return { hasError: true }; 
  }

  componentDidCatch(error, info) {
    console.error('Error in child component:', error, info);
  }

  render() {
    if (this.state.hasError) {
      return <div className="p-4 text-red-600">Something went wrong.</div>;
    }
    return this.props.children;
  }
}
