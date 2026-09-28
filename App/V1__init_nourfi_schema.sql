-- =============================================================================
-- Migration: V1__init_nourfi_schema.sql
-- Description: Création initiale du schéma de la base de données Nourfi_db
-- =============================================================================

-- 1. Table: User
CREATE TABLE IF NOT EXISTS public."User" (
    user_id VARCHAR NOT NULL,
    name TEXT,
    phone_number VARCHAR,
    email VARCHAR,
    created_at TIMESTAMP WITHOUT TIME ZONE,
    password VARCHAR,
    age INTEGER,
    is_verified BOOLEAN DEFAULT FALSE,
    CONSTRAINT "User_pkey" PRIMARY KEY (user_id)
);

-- 2. Table: Expert
CREATE TABLE IF NOT EXISTS public."Expert" (
    expert_id VARCHAR NOT NULL,
    name TEXT,
    email VARCHAR,
    password VARCHAR,
    specialization VARCHAR,
    reliability_score VARCHAR,
    CONSTRAINT "Expert_pkey" PRIMARY KEY (expert_id)
);

-- 3. Table: conversation
CREATE TABLE IF NOT EXISTS public.conversation (
    conv_id VARCHAR NOT NULL,
    user_id VARCHAR,
    title VARCHAR,
    created_at TIMESTAMP WITHOUT TIME ZONE,
    CONSTRAINT conversation_pkey PRIMARY KEY (conv_id),
    CONSTRAINT conversation_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."User"(user_id)
);

-- 4. Table: message
CREATE TABLE IF NOT EXISTS public.message (
    msg_id VARCHAR NOT NULL,
    conv_id VARCHAR,
    content TEXT,
    sender VARCHAR,
    "timestamp" TIMESTAMP WITHOUT TIME ZONE,
    CONSTRAINT message_pkey PRIMARY KEY (msg_id),
    CONSTRAINT message_conv_id_fkey FOREIGN KEY (conv_id) REFERENCES public.conversation(conv_id)
);

-- 5. Table: financial_profile
CREATE TABLE IF NOT EXISTS public.financial_profile (
    financial_id VARCHAR NOT NULL,
    user_id VARCHAR,
    family_stat VARCHAR,
    income VARCHAR,
    investments VARCHAR,
    job VARCHAR,
    postal_code VARCHAR,
    credit VARCHAR,
    down_payment VARCHAR,
    has_down_payment BOOLEAN,
    has_loans BOOLEAN,
    maingoal VARCHAR,
    knowledgelvl VARCHAR,
    ethical_priorities VARCHAR,
    networth VARCHAR,
    CONSTRAINT financial_profile_pkey PRIMARY KEY (financial_id),
    CONSTRAINT financial_profile_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."User"(user_id)
);

-- 6. Table: email_verification_code
CREATE TABLE IF NOT EXISTS public.email_verification_code (
    code_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    code VARCHAR NOT NULL,
    expires_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    CONSTRAINT email_verification_code_pkey PRIMARY KEY (code_id),
    CONSTRAINT email_verification_code_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."User"(user_id) ON DELETE CASCADE
);

-- 7. Table: password_reset_token
CREATE TABLE IF NOT EXISTS public.password_reset_token (
    token_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    token VARCHAR NOT NULL,
    expires_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    CONSTRAINT password_reset_token_pkey PRIMARY KEY (token_id),
    CONSTRAINT password_reset_token_token_key UNIQUE (token),
    CONSTRAINT password_reset_token_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."User"(user_id) ON DELETE CASCADE
);

-- 8. Table: source
CREATE TABLE IF NOT EXISTS public.source (
    source_id VARCHAR NOT NULL,
    type VARCHAR,
    url TEXT,
    validated_by VARCHAR,
    validated_at TIMESTAMP WITHOUT TIME ZONE,
    status VARCHAR DEFAULT 'pending'::character varying,
    CONSTRAINT source_pkey PRIMARY KEY (source_id),
    CONSTRAINT source_validated_by_fkey FOREIGN KEY (validated_by) REFERENCES public."Expert"(expert_id) ON DELETE SET NULL
);

-- 9. Table: message_source
CREATE TABLE IF NOT EXISTS public.message_source (
    source_id VARCHAR NOT NULL,
    msg_id VARCHAR NOT NULL,
    CONSTRAINT message_source_pkey PRIMARY KEY (source_id, msg_id),
    CONSTRAINT message_source_msg_id_fkey FOREIGN KEY (msg_id) REFERENCES public.message(msg_id),
    CONSTRAINT message_source_source_id_fkey FOREIGN KEY (source_id) REFERENCES public.source(source_id)
);