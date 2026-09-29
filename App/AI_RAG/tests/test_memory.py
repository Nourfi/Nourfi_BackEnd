import os
import time
 
from dotenv import load_dotenv
 
load_dotenv()
 
from App.AI_RAG.rag.pipeline import answer_question
 
DB_URL = (
    os.getenv("SUPABASE_DATABASE_URL")
    or os.getenv("SUPABASE_DB_URL")
    or os.getenv("DATABASE_URL")
)
 
PERSONA = "student"
PAUSE_SECONDS = 25  # respecte le plafond TPM Groq entre deux appels
 
 
def main():
    if not DB_URL:
        raise SystemExit("Aucune URL de base trouvée dans .env")
 
    # --- TOUR 1 : question autonome ---
    q1 = "C'est quoi la mourabaha ?"
    print(f"\n=== TOUR 1 ===\nQuestion : {q1}\n")
    a1, _ = answer_question(q1, DB_URL, persona_name=PERSONA)
    print(a1[:700])
 
    print(f"\n[pause {PAUSE_SECONDS}s pour éviter le rate limit Groq...]\n")
    time.sleep(PAUSE_SECONDS)
 
    # --- TOUR 2 : question qui n'a de sens QUE grâce au tour 1 ---
    history = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": a1},
    ]
    q2 = "Et pour un montant plus élevé, ça change quoi ?"
    print(f"=== TOUR 2 (avec mémoire) ===\nQuestion : {q2}\n")
    a2, _ = answer_question(q2, DB_URL, persona_name=PERSONA, recent_messages=history)
    print(a2[:900])
 
    # --- Verdict ---
    print("\n" + "=" * 60)
    print("CRITÈRE DE RÉUSSITE :")
    print("La réponse du tour 2 doit parler de mourabaha sans que")
    print("la question ne l'ait mentionnée. Si elle répond")
    print("'insufficient' ou parle d'un autre contrat, la mémoire")
    print("n'est pas transmise correctement.")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()
 