import { useEffect, useState } from "react";
import TopRightMenu from "./TopRightMenu";
import { MapPinHouse, Hospital } from "lucide-react";

export default function StrokeCenter() {
    const [location, setLocation] = useState(null);
    const [centers, setCenters] = useState([]);

    useEffect(() => {
        setLocation({ address: "서울시 동대문구 휘경동 18" });

        setCenters([
            {
                name: "서울대학교병원",
                address: "서울 종로구 대학로 10",
                distance: 1.2,
                phone: "02-760-1234",
                website: "https://www.snuh.org",
            },
            {
                name: "삼성서울병원",
                address: "서울 강남구 일원로 81",
                distance: 3.8,
                phone: "02-3410-0200",
                website: "https://www.samsunghospital.com",
            },
            {
                name: "세브란스병원",
                address: "서울 서대문구 연세로 50",
                distance: 2.5,
                phone: "02-2228-5807",
                website: "https://www.yonsei.ac.kr",
            },
            {
                name: "고려대학교병원",
                address: "서울 성북구 안암로 73",
                distance: 3.1,
                phone: "02-920-5114",
                website: "https://www.kumc.or.kr",
            },
            {
                name: "서울아산병원",
                address: "서울 송파구 올림픽로 43길 88",
                distance: 5.0,
                phone: "02-3010-0500",
                website: "https://www.amc.seoul.kr",
            },
            {
                name: "서울성모병원",
                address: "서울 서초구 반포대로 222",
                distance: 4.5,
                phone: "02-590-2114",
                website: "https://www.cmcseoul.or.kr",
            },
        ]);
    }, []);

    return (
        <div className="relative min-h-screen bg-white px-6 py-8">
            <TopRightMenu showLoginButton={false} />

            {/* 제목 및 설명 */}
            <div className="text-center mt-4">
                <h1 className="text-7xl font-bold text-blue-600 mt-10 mb-8">주변 뇌졸중 센터</h1>
                <p className="inline-block bg-gray-100 px-12 py-1.5 text-[1.6rem] rounded-full text-black">
                    현재 위치를 기반으로 가까운 센터를 추천합니다.
                </p>
            </div>

            {/* 현재 위치 */}
            {location?.address && (
                <div className="mt-8 text-center flex items-center justify-center text-gray-700 gap-2">
                    <MapPinHouse className="w-7 h-7 text-black" />
                    <span className="underline text-black text-2xl">{location.address}</span>
                </div>
            )}

            {/* 병원 리스트 */}
            <div className="mt-14 ml-24 mr-24 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-16 gap-y-8">
                {centers.map((center, index) => (
                    <div
                        key={index}
                        className="bg-gray-50 rounded-2xl p-4 shadow-md hover:shadow-lg transition flex flex-col"
                    >
                        {/* 아이콘 + 병원 이름 */}
                        <div className="flex items-center gap-4 mb-4">
                            <div className="bg-white rounded-full p-2 shadow-md w-20 h-20 flex items-center justify-center">
                                <Hospital className="w-11 h-11 text-black" strokeWidth={1.5} />
                            </div>
                            <h3 className="text-[1.7rem] font-bold text-blue-600">{center.name}</h3>
                        </div>

                        {/* 항목 리스트 */}
                        <div className="space-y-2 text-lg text-black">
                            <div className="flex gap-3">
                                <span className="font-semibold w-24 shrink-0">주소:</span>
                                <span>{center.address}</span>
                            </div>
                            <div className="flex gap-3">
                                <span className="font-semibold w-24 shrink-0">거리:</span>
                                <span>{center.distance} km</span>
                            </div>
                            <div className="flex gap-3">
                                <span className="font-semibold w-24 shrink-0">연락처:</span>
                                <span>{center.phone}</span>
                            </div>
                            <div className="flex gap-3">
                                <span className="font-semibold w-24 shrink-0">홈페이지:</span>
                                <a
                                    href={center.website}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-black underline text-lg hover:font-semibold"
                                >
                                    {center.website}
                                </a>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
