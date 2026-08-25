
from utils.agents import video_agent_app
from langchain_core.messages import HumanMessage, SystemMessage

def main():
    print("--- Video Analiz AI Sistemine Hoş Geldiniz ---")
    print("Çıkmak için 'q' veya 'quit' yazın.\n")
    
    # 1. Dış hafıza: sadece kullanıcı girişleri ve çalışma akışının görünüm/izleme amaçlı takibi.
    chat_history = []

    while True:
        user_input = input("\nSen: ")
        if user_input.lower() in ['q', 'quit', 'çıkış']:
            break

        if not user_input.strip():
            continue

        # 2. Kullanıcı mesajını dış hafızaya ekle.
        chat_history.append(HumanMessage(content=user_input))

        # 3. LangGraph state'i, gerçek akış için gerekli alanları içerir.
        #    video_path gibi değerler araçlara runtime parametre olarak verilir; state içinde sabit tutmak gereksizdir.
        initial_state = {
            "user_query": user_input,
            "plan": "",
            "messages": chat_history,
            "feedback": "",
            "final_answer": "",
            "replan_reason": "",
            "replan_count": 0,
        }

        print("\n[Sistem Düşünüyor...]")
        
        final_state = None
        for event in video_agent_app.stream(initial_state, {"recursion_limit": 105}):
            for node_name, node_state in event.items():
                final_state = node_state
                
                print(f"\n--- [AKTİF NODE: {node_name.upper()}] ---")
                if node_name == "planner":
                    print(f"Oluşturulan Plan:\n{node_state.get('plan')}")
                elif node_name == "executor":
                    last_msg = node_state.get("messages")[-1]
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tool in last_msg.tool_calls:
                            print(f"🔧 Araca Başvuruluyor: {tool['name']}")
                    else:
                        print("İşlemci yorum yapıyor...")
                elif node_name == "tools":
                    print("✅ Araç (Tool) çalıştı ve sonuç döndürdü.")
                elif node_name == "reviewer":
                    feedback = node_state.get("feedback")
                    final_answer = node_state.get("final_answer")
                    if feedback:
                        print(f"❌ Denetleyici Eksik Buldu: {feedback}")
                    elif final_answer:
                        print(f"\n🤖 NİHAİ CEVAP:\n{final_answer}")
                    else:
                        print("✅ Denetleyici Onayladı.")

        # 4. DÖNGÜ SONUNDA HAFIZAYI GÜNCELLİYORUZ
        if final_state and "messages" in final_state:
            # LangGraph'ın eklediği tool cevapları ve AI cevaplarıyla birlikte hafızayı kaydediyoruz
            chat_history = final_state["messages"] 

if __name__ == "__main__":
    main()