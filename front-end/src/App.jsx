import { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Carousel from "./components/Carousel";
import LoginModal from "./components/LoginModal";
import SignupModal from "./components/SignupModal";
import StrokeCenter from "./components/StrokeCenter";
import TestCarousel from "./components/TestCarousel";
import TopRightMenu from "./components/TopRightMenu";

function App() {
    const [isLoginOpen, setIsLoginOpen] = useState(false);
    const [isSignupOpen, setIsSignupOpen] = useState(false);
    const [isLoggedIn, setIsLoggedIn] = useState(false); // 로그인 상태 추가

    return (
        <Router>
            <Routes>
                <Route
                    path="/"
                    element={
                        <div className="w-full h-screen relative">
                            <Carousel
                                onLoginClick={() => setIsLoginOpen(true)}
                                isLoggedIn={isLoggedIn}
                                setIsLoggedIn={setIsLoggedIn}
                            />
                            <TopRightMenu
                                onLoginClick={() => setIsLoginOpen(true)}
                                isLoggedIn={isLoggedIn}
                                setIsLoggedIn={setIsLoggedIn}
                            />
                            {isLoginOpen && (
                                <LoginModal
                                    onClose={() => setIsLoginOpen(false)}
                                    onSignupOpen={() => {
                                        setIsLoginOpen(false);
                                        setIsSignupOpen(true);
                                    }}
                                    onLoginSuccess={() => {
                                        setIsLoggedIn(true);
                                        setIsLoginOpen(false);
                                    }}
                                />
                            )}
                            {isSignupOpen && (
                                <SignupModal
                                    onClose={() => setIsSignupOpen(false)}
                                    onLoginOpen={() => {
                                        setIsSignupOpen(false);
                                        setIsLoginOpen(true);
                                    }}
                                />
                            )}
                        </div>
                    }
                />
                <Route path="/stroke-center" element={<StrokeCenter />} />
                <Route path="/test" element={<TestCarousel />} />
            </Routes>
        </Router>
    );
}

export default App;