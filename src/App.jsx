import ErrorBoundary from './ErrorBoundary';
import Chat from './Chat';
import './index.css';

export default function App() {
  return (
    <ErrorBoundary>
      <Chat />
    </ErrorBoundary>
  );
}
