"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[835],{57796:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>g,contentTitle:()=>c,default:()=>y,frontMatter:()=>p,metadata:()=>u,toc:()=>b});var i=a(87462),s=(a(67294),a(3905)),n=a(26389),o=a(94891),r=(a(75190),a(47507)),l=a(7813),d=a(63303),m=(a(75035),a(85162));const p={id:"api-encode",title:"encode",description:"Generates a vector representation of text or images using the specified embedding model. If the model does not exist or has not been loaded, an error will be returned for that particular data item. <br/><br/>It is recommended that multiple pieces of content are specified in a single request to take advantage of batching, especially when running under with a GPU accelerated <a href='https://onnxruntime.ai/docs/execution-providers/' target='_blank'>ExecutionProvider.</a> Recommended sizes can be found at <a href='/embedds/supported-models' target='_blank'>embedds/model-details</a> and embedds will attempt to split and batch encoding requests.<br/>----<br/><b>Not all models support [text, instructions, image, image_uri]</b> a breakdown of present support is outlined below:<ul><li>ModelClass_INSTRUCTOR - requires: {text, instructions}</li><li>ModelClass_CLIP - any_of: {text, image_uri, image}</li></ul>",sidebar_label:"encode",hide_title:!0,hide_table_of_contents:!0,api:{description:"Generates a vector representation of text or images using the specified embedding model. If the model does not exist or has not been loaded, an error will be returned for that particular data item. <br/><br/>It is recommended that multiple pieces of content are specified in a single request to take advantage of batching, especially when running under with a GPU accelerated <a href='https://onnxruntime.ai/docs/execution-providers/' target='_blank'>ExecutionProvider.</a> Recommended sizes can be found at <a href='/embedds/supported-models' target='_blank'>embedds/model-details</a> and embedds will attempt to split and batch encoding requests.<br/>----<br/><b>Not all models support [text, instructions, image, image_uri]</b> a breakdown of present support is outlined below:<ul><li>ModelClass_INSTRUCTOR - requires: {text, instructions}</li><li>ModelClass_CLIP - any_of: {text, image_uri, image}</li></ul>",operationId:"Api_Encode",responses:{200:{description:"A successful response.",content:{"application/json":{schema:{type:"object",properties:{results:{type:"array",items:{type:"object",properties:{err_message:{type:"string",description:"explanation for why the content could not be encoded"},embedding:{type:"array",items:{type:"number",format:"float"},description:"embedding representation of the the content"}},title:"apiEncodeResult"},description:"list of embedding results corresponding to the ordered content submitted"}},title:"apiEncodeResponse"}}}},default:{description:"An unexpected error response.",content:{"application/json":{schema:{type:"object",properties:{code:{type:"integer",format:"int32"},message:{type:"string"},details:{type:"array",items:{type:"object",properties:{"@type":{type:"string"}},additionalProperties:{},title:"protobufAny"}}},title:"rpcStatus"}}}}},requestBody:{content:{"application/json":{schema:{type:"object",example:{data:[{model_class:"ModelClass_INSTRUCTOR",model_name:"INSTRUCTOR_LARGE",text:["3D ActionSLAM: wearable person tracking ...","Inside Gohar World and the Fine, Fantastical Art"],instructions:["Represent the Science title:","Represent the Magazine title:"]}]},properties:{data:{type:"array",items:{type:"object",properties:{model_class:{type:"string",enum:["ModelClass_Unknown","ModelClass_CLIP","ModelClass_INSTRUCTOR"],default:"ModelClass_Unknown",title:"apiModelClass"},model_name:{type:"string",description:"specific name of the model to apply the encoding transformation"},text:{type:"array",items:{type:"string"},description:"repeated text to encode"},instructions:{type:"array",items:{type:"string"},description:"a list of instructions to pass to ```INSTRUCTOR``` based models"},image:{type:"array",items:{type:"string",format:"byte"},description:"repeated raw jpeg bytes"},image_uri:{type:"array",items:{type:"string"},description:"repeated uris to fetch image data from"}},description:"Minimal encoding unit associating a piece of content [text, image, image_uri] with a selected model",required:["model"],title:"apiEncodeItem"},description:"repeated data items to generate encodings for"}},title:"apiEncodeRequest"}}},required:!0},tags:["Api"],method:"post",path:"/encode",jsonRequestBodyExample:{data:[{model_class:"ModelClass_INSTRUCTOR",model_name:"INSTRUCTOR_LARGE",text:["3D ActionSLAM: wearable person tracking ...","Inside Gohar World and the Fine, Fantastical Art"],instructions:["Represent the Science title:","Represent the Magazine title:"]}]},info:{title:"embedds",version:"1.0.0",contact:{name:"anansi",url:"https://github.com/infrawhispers/anansi",email:"infrawhispers@proton.me"},license:{name:"Apache 2.0 License",url:"https://github.com/infrawhispers/anansi/blob/main/LICENSE"}},postman:{name:"encode",description:{content:"Generates a vector representation of text or images using the specified embedding model. If the model does not exist or has not been loaded, an error will be returned for that particular data item. <br/><br/>It is recommended that multiple pieces of content are specified in a single request to take advantage of batching, especially when running under with a GPU accelerated <a href='https://onnxruntime.ai/docs/execution-providers/' target='_blank'>ExecutionProvider.</a> Recommended sizes can be found at <a href='/embedds/supported-models' target='_blank'>embedds/model-details</a> and embedds will attempt to split and batch encoding requests.<br/>----<br/><b>Not all models support [text, instructions, image, image_uri]</b> a breakdown of present support is outlined below:<ul><li>ModelClass_INSTRUCTOR - requires: {text, instructions}</li><li>ModelClass_CLIP - any_of: {text, image_uri, image}</li></ul>",type:"text/plain"},url:{path:["encode"],host:["{{baseUrl}}"],query:[],variable:[]},header:[{key:"Content-Type",value:"application/json"},{key:"Accept",value:"application/json"}],method:"POST",body:{mode:"raw",raw:'""',options:{raw:{language:"json"}}}}},sidebar_class_name:"post api-method",info_path:"swagger-api/embedds",custom_edit_url:null},c=void 0,u={unversionedId:"swagger-api/api-encode",id:"swagger-api/api-encode",title:"encode",description:"Generates a vector representation of text or images using the specified embedding model. If the model does not exist or has not been loaded, an error will be returned for that particular data item. <br/><br/>It is recommended that multiple pieces of content are specified in a single request to take advantage of batching, especially when running under with a GPU accelerated <a href='https://onnxruntime.ai/docs/execution-providers/' target='_blank'>ExecutionProvider.</a> Recommended sizes can be found at <a href='/embedds/supported-models' target='_blank'>embedds/model-details</a> and embedds will attempt to split and batch encoding requests.<br/>----<br/><b>Not all models support [text, instructions, image, image_uri]</b> a breakdown of present support is outlined below:<ul><li>ModelClass_INSTRUCTOR - requires: {text, instructions}</li><li>ModelClass_CLIP - any_of: {text, image_uri, image}</li></ul>",source:"@site/docs/swagger-api/api-encode.api.mdx",sourceDirName:"swagger-api",slug:"/swagger-api/api-encode",permalink:"/anansi/swagger-api/api-encode",draft:!1,editUrl:null,tags:[],version:"current",frontMatter:{id:"api-encode",title:"encode",description:"Generates a vector representation of text or images using the specified embedding model. If the model does not exist or has not been loaded, an error will be returned for that particular data item. <br/><br/>It is recommended that multiple pieces of content are specified in a single request to take advantage of batching, especially when running under with a GPU accelerated <a href='https://onnxruntime.ai/docs/execution-providers/' target='_blank'>ExecutionProvider.</a> Recommended sizes can be found at <a href='/embedds/supported-models' target='_blank'>embedds/model-details</a> and embedds will attempt to split and batch encoding requests.<br/>----<br/><b>Not all models support [text, instructions, image, image_uri]</b> a breakdown of present support is outlined below:<ul><li>ModelClass_INSTRUCTOR - requires: {text, instructions}</li><li>ModelClass_CLIP - any_of: {text, image_uri, image}</li></ul>",sidebar_label:"encode",hide_title:!0,hide_table_of_contents:!0,api:{description:"Generates a vector representation of text or images using the specified embedding model. If the model does not exist or has not been loaded, an error will be returned for that particular data item. <br/><br/>It is recommended that multiple pieces of content are specified in a single request to take advantage of batching, especially when running under with a GPU accelerated <a href='https://onnxruntime.ai/docs/execution-providers/' target='_blank'>ExecutionProvider.</a> Recommended sizes can be found at <a href='/embedds/supported-models' target='_blank'>embedds/model-details</a> and embedds will attempt to split and batch encoding requests.<br/>----<br/><b>Not all models support [text, instructions, image, image_uri]</b> a breakdown of present support is outlined below:<ul><li>ModelClass_INSTRUCTOR - requires: {text, instructions}</li><li>ModelClass_CLIP - any_of: {text, image_uri, image}</li></ul>",operationId:"Api_Encode",responses:{200:{description:"A successful response.",content:{"application/json":{schema:{type:"object",properties:{results:{type:"array",items:{type:"object",properties:{err_message:{type:"string",description:"explanation for why the content could not be encoded"},embedding:{type:"array",items:{type:"number",format:"float"},description:"embedding representation of the the content"}},title:"apiEncodeResult"},description:"list of embedding results corresponding to the ordered content submitted"}},title:"apiEncodeResponse"}}}},default:{description:"An unexpected error response.",content:{"application/json":{schema:{type:"object",properties:{code:{type:"integer",format:"int32"},message:{type:"string"},details:{type:"array",items:{type:"object",properties:{"@type":{type:"string"}},additionalProperties:{},title:"protobufAny"}}},title:"rpcStatus"}}}}},requestBody:{content:{"application/json":{schema:{type:"object",example:{data:[{model_class:"ModelClass_INSTRUCTOR",model_name:"INSTRUCTOR_LARGE",text:["3D ActionSLAM: wearable person tracking ...","Inside Gohar World and the Fine, Fantastical Art"],instructions:["Represent the Science title:","Represent the Magazine title:"]}]},properties:{data:{type:"array",items:{type:"object",properties:{model_class:{type:"string",enum:["ModelClass_Unknown","ModelClass_CLIP","ModelClass_INSTRUCTOR"],default:"ModelClass_Unknown",title:"apiModelClass"},model_name:{type:"string",description:"specific name of the model to apply the encoding transformation"},text:{type:"array",items:{type:"string"},description:"repeated text to encode"},instructions:{type:"array",items:{type:"string"},description:"a list of instructions to pass to ```INSTRUCTOR``` based models"},image:{type:"array",items:{type:"string",format:"byte"},description:"repeated raw jpeg bytes"},image_uri:{type:"array",items:{type:"string"},description:"repeated uris to fetch image data from"}},description:"Minimal encoding unit associating a piece of content [text, image, image_uri] with a selected model",required:["model"],title:"apiEncodeItem"},description:"repeated data items to generate encodings for"}},title:"apiEncodeRequest"}}},required:!0},tags:["Api"],method:"post",path:"/encode",jsonRequestBodyExample:{data:[{model_class:"ModelClass_INSTRUCTOR",model_name:"INSTRUCTOR_LARGE",text:["3D ActionSLAM: wearable person tracking ...","Inside Gohar World and the Fine, Fantastical Art"],instructions:["Represent the Science title:","Represent the Magazine title:"]}]},info:{title:"embedds",version:"1.0.0",contact:{name:"anansi",url:"https://github.com/infrawhispers/anansi",email:"infrawhispers@proton.me"},license:{name:"Apache 2.0 License",url:"https://github.com/infrawhispers/anansi/blob/main/LICENSE"}},postman:{name:"encode",description:{content:"Generates a vector representation of text or images using the specified embedding model. If the model does not exist or has not been loaded, an error will be returned for that particular data item. <br/><br/>It is recommended that multiple pieces of content are specified in a single request to take advantage of batching, especially when running under with a GPU accelerated <a href='https://onnxruntime.ai/docs/execution-providers/' target='_blank'>ExecutionProvider.</a> Recommended sizes can be found at <a href='/embedds/supported-models' target='_blank'>embedds/model-details</a> and embedds will attempt to split and batch encoding requests.<br/>----<br/><b>Not all models support [text, instructions, image, image_uri]</b> a breakdown of present support is outlined below:<ul><li>ModelClass_INSTRUCTOR - requires: {text, instructions}</li><li>ModelClass_CLIP - any_of: {text, image_uri, image}</li></ul>",type:"text/plain"},url:{path:["encode"],host:["{{baseUrl}}"],query:[],variable:[]},header:[{key:"Content-Type",value:"application/json"},{key:"Accept",value:"application/json"}],method:"POST",body:{mode:"raw",raw:'""',options:{raw:{language:"json"}}}}},sidebar_class_name:"post api-method",info_path:"swagger-api/embedds",custom_edit_url:null},sidebar:"tutorialSidebar",previous:{title:"Introduction",permalink:"/anansi/swagger-api/embedds"},next:{title:"initialize_models",permalink:"/anansi/swagger-api/api-initialize-model"}},g={},b=[{value:"encode",id:"encode",level:2}],h={toc:b},f="wrapper";function y(e){let{components:t,...a}=e;return(0,s.kt)(f,(0,i.Z)({},h,a,{components:t,mdxType:"MDXLayout"}),(0,s.kt)("h2",{id:"encode"},"encode"),(0,s.kt)("p",null,"Generates a vector representation of text or images using the specified embedding model. If the model does not exist or has not been loaded, an error will be returned for that particular data item. ",(0,s.kt)("br",null),(0,s.kt)("br",null),"It is recommended that multiple pieces of content are specified in a single request to take advantage of batching, especially when running under with a GPU accelerated ",(0,s.kt)("a",{href:"https://onnxruntime.ai/docs/execution-providers/",target:"_blank"},"ExecutionProvider.")," Recommended sizes can be found at ",(0,s.kt)("a",{href:"/embedds/supported-models",target:"_blank"},"embedds/model-details")," and embedds will attempt to split and batch encoding requests.",(0,s.kt)("br",null),"----",(0,s.kt)("br",null),(0,s.kt)("b",null,"Not all models support ","[text, instructions, image, image_uri]")," a breakdown of present support is outlined below:",(0,s.kt)("ul",null,(0,s.kt)("li",null,"ModelClass_INSTRUCTOR - requires: {text, instructions}"),(0,s.kt)("li",null,"ModelClass_CLIP - any_of: {text, image_uri, image}"))),(0,s.kt)(o.Z,{mdxType:"MimeTabs"},(0,s.kt)(m.Z,{label:"application/json",value:"application/json-schema",mdxType:"TabItem"},(0,s.kt)("details",{style:{},"data-collapsed":!1,open:!0},(0,s.kt)("summary",{style:{textAlign:"left"}},(0,s.kt)("strong",null,"Request Body"),(0,s.kt)("strong",{style:{fontSize:"var(--ifm-code-font-size)",color:"var(--openapi-required)"}}," required")),(0,s.kt)("div",{style:{textAlign:"left",marginLeft:"1rem"}}),(0,s.kt)("ul",{style:{marginLeft:"1rem"}},(0,s.kt)(l.Z,{collapsible:!0,className:"schemaItem",mdxType:"SchemaItem"},(0,s.kt)("details",{style:{}},(0,s.kt)("summary",{style:{}},(0,s.kt)("strong",null,"data"),(0,s.kt)("span",{style:{opacity:"0.6"}}," object[]")),(0,s.kt)("div",{style:{marginLeft:"1rem"}},(0,s.kt)("div",{style:{marginTop:".5rem",marginBottom:".5rem"}},(0,s.kt)("p",null,"repeated data items to generate encodings for")),(0,s.kt)("li",null,(0,s.kt)("div",{style:{fontSize:"var(--ifm-code-font-size)",opacity:"0.6",marginLeft:"-.5rem",paddingBottom:".5rem"}},"Array [")),(0,s.kt)(l.Z,{collapsible:!1,name:"model_class",required:!1,schemaName:"apiModelClass",qualifierMessage:"**Possible values:** [`ModelClass_Unknown`, `ModelClass_CLIP`, `ModelClass_INSTRUCTOR`]",schema:{type:"string",enum:["ModelClass_Unknown","ModelClass_CLIP","ModelClass_INSTRUCTOR"],default:"ModelClass_Unknown",title:"apiModelClass"},mdxType:"SchemaItem"}),(0,s.kt)(l.Z,{collapsible:!1,name:"model_name",required:!1,schemaName:"string",qualifierMessage:void 0,schema:{type:"string",description:"specific name of the model to apply the encoding transformation"},mdxType:"SchemaItem"}),(0,s.kt)(l.Z,{collapsible:!1,name:"text",required:!1,schemaName:"string[]",qualifierMessage:void 0,schema:{type:"array",items:{type:"string"},description:"repeated text to encode"},mdxType:"SchemaItem"}),(0,s.kt)(l.Z,{collapsible:!1,name:"instructions",required:!1,schemaName:"string[]",qualifierMessage:void 0,schema:{type:"array",items:{type:"string"},description:"a list of instructions to pass to ```INSTRUCTOR``` based models"},mdxType:"SchemaItem"}),(0,s.kt)(l.Z,{collapsible:!1,name:"image",required:!1,schemaName:"byte[]",qualifierMessage:void 0,schema:{type:"array",items:{type:"string",format:"byte"},description:"repeated raw jpeg bytes"},mdxType:"SchemaItem"}),(0,s.kt)(l.Z,{collapsible:!1,name:"image_uri",required:!1,schemaName:"string[]",qualifierMessage:void 0,schema:{type:"array",items:{type:"string"},description:"repeated uris to fetch image data from"},mdxType:"SchemaItem"}),(0,s.kt)("li",null,(0,s.kt)("div",{style:{fontSize:"var(--ifm-code-font-size)",opacity:"0.6",marginLeft:"-.5rem"}},"]"))))))))),(0,s.kt)("div",null,(0,s.kt)(n.Z,{mdxType:"ApiTabs"},(0,s.kt)(m.Z,{label:"200",value:"200",mdxType:"TabItem"},(0,s.kt)("div",null,(0,s.kt)("p",null,"A successful response.")),(0,s.kt)("div",null,(0,s.kt)(o.Z,{schemaType:"response",mdxType:"MimeTabs"},(0,s.kt)(m.Z,{label:"application/json",value:"application/json",mdxType:"TabItem"},(0,s.kt)(d.Z,{mdxType:"SchemaTabs"},(0,s.kt)(m.Z,{label:"Schema",value:"Schema",mdxType:"TabItem"},(0,s.kt)("details",{style:{},"data-collapsed":!1,open:!0},(0,s.kt)("summary",{style:{textAlign:"left"}},(0,s.kt)("strong",null,"Schema")),(0,s.kt)("div",{style:{textAlign:"left",marginLeft:"1rem"}}),(0,s.kt)("ul",{style:{marginLeft:"1rem"}},(0,s.kt)(l.Z,{collapsible:!0,className:"schemaItem",mdxType:"SchemaItem"},(0,s.kt)("details",{style:{}},(0,s.kt)("summary",{style:{}},(0,s.kt)("strong",null,"results"),(0,s.kt)("span",{style:{opacity:"0.6"}}," object[]")),(0,s.kt)("div",{style:{marginLeft:"1rem"}},(0,s.kt)("div",{style:{marginTop:".5rem",marginBottom:".5rem"}},(0,s.kt)("p",null,"list of embedding results corresponding to the ordered content submitted")),(0,s.kt)("li",null,(0,s.kt)("div",{style:{fontSize:"var(--ifm-code-font-size)",opacity:"0.6",marginLeft:"-.5rem",paddingBottom:".5rem"}},"Array [")),(0,s.kt)(l.Z,{collapsible:!1,name:"err_message",required:!1,schemaName:"string",qualifierMessage:void 0,schema:{type:"string",description:"explanation for why the content could not be encoded"},mdxType:"SchemaItem"}),(0,s.kt)(l.Z,{collapsible:!1,name:"embedding",required:!1,schemaName:"float[]",qualifierMessage:void 0,schema:{type:"array",items:{type:"number",format:"float"},description:"embedding representation of the the content"},mdxType:"SchemaItem"}),(0,s.kt)("li",null,(0,s.kt)("div",{style:{fontSize:"var(--ifm-code-font-size)",opacity:"0.6",marginLeft:"-.5rem"}},"]")))))))),(0,s.kt)(m.Z,{label:"Example (from schema)",value:"Example (from schema)",mdxType:"TabItem"},(0,s.kt)(r.Z,{responseExample:'{\n  "results": [\n    {\n      "err_message": "string",\n      "embedding": [\n        0\n      ]\n    }\n  ]\n}',language:"json",mdxType:"ResponseSamples"}))))))),(0,s.kt)(m.Z,{label:"default",value:"default",mdxType:"TabItem"},(0,s.kt)("div",null,(0,s.kt)("p",null,"An unexpected error response.")),(0,s.kt)("div",null,(0,s.kt)(o.Z,{schemaType:"response",mdxType:"MimeTabs"},(0,s.kt)(m.Z,{label:"application/json",value:"application/json",mdxType:"TabItem"},(0,s.kt)(d.Z,{mdxType:"SchemaTabs"},(0,s.kt)(m.Z,{label:"Schema",value:"Schema",mdxType:"TabItem"},(0,s.kt)("details",{style:{},"data-collapsed":!1,open:!0},(0,s.kt)("summary",{style:{textAlign:"left"}},(0,s.kt)("strong",null,"Schema")),(0,s.kt)("div",{style:{textAlign:"left",marginLeft:"1rem"}}),(0,s.kt)("ul",{style:{marginLeft:"1rem"}},(0,s.kt)(l.Z,{collapsible:!1,name:"code",required:!1,schemaName:"int32",qualifierMessage:void 0,schema:{type:"integer",format:"int32"},mdxType:"SchemaItem"}),(0,s.kt)(l.Z,{collapsible:!1,name:"message",required:!1,schemaName:"string",qualifierMessage:void 0,schema:{type:"string"},mdxType:"SchemaItem"}),(0,s.kt)(l.Z,{collapsible:!0,className:"schemaItem",mdxType:"SchemaItem"},(0,s.kt)("details",{style:{}},(0,s.kt)("summary",{style:{}},(0,s.kt)("strong",null,"details"),(0,s.kt)("span",{style:{opacity:"0.6"}}," object[]")),(0,s.kt)("div",{style:{marginLeft:"1rem"}},(0,s.kt)("li",null,(0,s.kt)("div",{style:{fontSize:"var(--ifm-code-font-size)",opacity:"0.6",marginLeft:"-.5rem",paddingBottom:".5rem"}},"Array [")),(0,s.kt)(l.Z,{collapsible:!1,name:"@type",required:!1,schemaName:"string",qualifierMessage:void 0,schema:{type:"string"},mdxType:"SchemaItem"}),(0,s.kt)("li",null,(0,s.kt)("div",{style:{fontSize:"var(--ifm-code-font-size)",opacity:"0.6",marginLeft:"-.5rem"}},"]")))))))),(0,s.kt)(m.Z,{label:"Example (from schema)",value:"Example (from schema)",mdxType:"TabItem"},(0,s.kt)(r.Z,{responseExample:'{\n  "code": 0,\n  "message": "string",\n  "details": [\n    {\n      "@type": "string"\n    }\n  ]\n}',language:"json",mdxType:"ResponseSamples"}))))))))))}y.isMDXComponent=!0}}]);