import { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Carousel from "./components/Carousel";
import LoginModal from "./components/LoginModal";
import SignupModal from "./components/SignupModal";
import StrokeCenter from "./components/StrokeCenter";
import TestCarousel from "./components/TestCarousel";
import TopRightMenu from "./components/TopRightMenu";
import FaceMeasure from "./components/FaceMeasure";
import ArmMeasure from "./components/ArmMeasure.jsx";
import http from "./lib/http";

function App() {
    const [isLoginOpen, setIsLoginOpen] = useState(false);
    const [isSignupOpen, setIsSignupOpen] = useState(false);
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [loading, setLoading] = useState(true);

    // ★ 앱 시작 시 쿠키로 세션 상태 동기화
    useEffect(() => {
        (async () => {
            try {
                await http.get("/api/v1/auth/me");
                setIsLoggedIn(true);
            } catch {
                setIsLoggedIn(false);
            } finally {
                setLoading(false);
            }
        })();
    }, []);

    if (loading) return <div />;

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
                <Route path="/measure/face" element={<FaceMeasure />} />
                <Route path="/measure/arm" element={<ArmMeasure />} />
            </Routes>
        </Router>
    );
}

export default App;
